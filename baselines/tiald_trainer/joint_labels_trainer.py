"""
Joint Multi-Label Training for TiALD
"""

import json
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

# Monkey-patch the simpletransformers module to use the XLMRobertaTokenizerFast
import simpletransformers.classification.multi_label_classification_model as ml_module
import torch
import wandb
from datasets import load_dataset
from transformers import XLMRobertaTokenizerFast

ml_module.XLMRobertaTokenizer = XLMRobertaTokenizerFast

from simpletransformers.classification import MultiLabelClassificationModel  # noqa: E402

from .utils import TASK_INFO, compute_task_metrics  # noqa: E402

MODEL_TYPES = (
    "albert",
    "bert",
    "bertweet",
    "camembert",
    "distilbert",
    "electra",
    "flaubert",
    "herbert",
    "layoutlm",
    "layoutlmv2",
    "mobilebert",
    "rembert",
    "roberta",
    "xlm",
    "xlmroberta",
    "xlnet",
)


class TiALDMultiTaskModel(MultiLabelClassificationModel):
    """
    Subclass to override loss computation:
      - separate heads for each task
      - curriculum learning
    """

    def __init__(
        self,
        model_type,
        model_name,
        num_labels=None,
        pos_weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        abuse_coef: float = 1.0,
        sentiment_coef: float = 0.5,
        topic_coef: float = 0.33,
        tiald_curriculum_loss: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_type=model_type,
            model_name=model_name,
            num_labels=num_labels,
            pos_weight=pos_weight,
            args=args,
            use_cuda=use_cuda,
            cuda_device=cuda_device,
            **kwargs,
        )
        self.tiald_global_step = 0
        self.tiald_total_training_steps = kwargs.get("tiald_total_training_steps", 0)
        # Static task coefficients
        self.tiald_abuse_coef = abuse_coef
        self.tiald_sentiment_coef = sentiment_coef
        self.tiald_topic_coef = topic_coef
        # Curriculum phase durations
        self.tiald_curriculum_loss = tiald_curriculum_loss


class TiALDJointLabelsTrainer:
    def __init__(
        self,
        input_model_name: str,
        model_type: str,
        output_model_name: str,
        dataset_name: str,
        hf_token: str = None,
        output_dir: str = "./results",
        max_length: int = 256,
        batch_size: int = 8,
        lr: float = 3e-5,
        epochs: int = 7,
        gradient_accumulation_steps: int = 1,
        curriculum_loss: bool = False,
        report_to: str = "none",
    ):
        self.input_model_name = input_model_name
        self.output_model_name = output_model_name
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, output_model_name)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.max_length = max_length
        self.model_type = model_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.curriculum_loss = curriculum_loss
        self.hf_token = hf_token

        self.use_cuda = torch.cuda.is_available()
        print(f"Using device: {'cuda' if self.use_cuda else 'cpu'}")
        self.report_to = report_to

        # Label definitions
        self.abusiveness_labels = TASK_INFO["abusiveness"]["label_list"]
        self.topic_labels = TASK_INFO["topic"]["label_list"]
        self.sentiment_labels = TASK_INFO["sentiment"]["label_list"]
        self.num_labels = len(self.abusiveness_labels) + len(self.topic_labels) + len(self.sentiment_labels)

        # Load dataset
        self.dataset = None
        self.train_df = None
        self.eval_df = None
        self.test_df = None
        self.model = None

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_data_and_model()

    def _setup_data_and_model(self):
        """Setup reporting and load the dataset"""
        if self.report_to == "wandb":
            wandb.init(
                project="tiald-joint-labels",
                name=f"{self.output_model_name}-{self.timestamp}",
                config={
                    "task": "joint-labels",
                    "input_model": self.input_model_name,
                    "output_model": self.output_model_name,
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "max_length": self.max_length,
                    "epochs": self.epochs,
                },
            )
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading dataset {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name, token=self.hf_token)
        print("Loaded dataset:", self.dataset)
        # Convert to DataFrames and add labels
        self.train_df = self.dataset["train"].to_pandas()
        self.add_labels_field(self.train_df)
        self.eval_df = self.dataset["validation"].to_pandas()
        self.add_labels_field(self.eval_df)
        self.test_df = self.dataset["test"].to_pandas()
        self.add_labels_field(self.test_df)

    def add_labels_field(self, df: pd.DataFrame) -> None:
        """Add multi-label encoding to DataFrame"""
        labels = []
        for abusiveness, topic, sentiment in zip(df.abusiveness.tolist(), df.topic.tolist(), df.sentiment.tolist()):
            _lbls = [1 if abusiveness == t else 0 for t in self.abusiveness_labels]
            _lbls += [1 if topic == t else 0 for t in self.topic_labels]
            _lbls += [1 if sentiment == t else 0 for t in self.sentiment_labels]
            labels.append(_lbls)
        df["labels"] = labels

        # Use clean comment + video title as context
        # input_texts = []
        # for vide_title, comment in zip(df.video_title.tolist(), df.clean.tolist()):
        #     vide_title = " ".join(vide_title.split()).strip()
        #     comment = comment.strip()
        #     input_texts.append(f"Video Title: {vide_title}; Comment: {comment}")
        # df["text"] = input_texts

        # Use clean comment only
        df["text"] = df["comment_clean"].apply(lambda x: x.replace("\n", " "))

    def get_true_labels(self, model_outputs: List) -> Tuple[List, List, List]:
        """Extract predicted labels from model outputs

        Returns:
            _types: list of predicted abusiveness labels
            _sentiments: list of predicted sentiment labels
            _topics: list of predicted topic labels
        """
        _types, _topics, _sentiments = list(), list(), list()
        for labels in model_outputs:
            _type = self.abusiveness_labels[np.argmax(labels[0:2])]
            _sentiment = self.sentiment_labels[np.argmax(labels[7:])]
            _topic = self.topic_labels[np.argmax(labels[2:7])]

            _types.append(_type)
            _topics.append(_topic)
            _sentiments.append(_sentiment)
        return _types, _sentiments, _topics

    def train(self):
        """Train the multi-label model"""
        model_args = {
            "output_dir": self.output_dir,
            "overwrite_output_dir": True,
            "max_seq_length": self.max_length,
            # Batch & Accumulation
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            # Learning Rate & Scheduler
            "learning_rate": self.lr,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "linear",  # or "cosine"
            # Regularization & Optimization
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            # Gradient Clipping
            "max_grad_norm": 1.0,
            # Evaluation & Checkpoints
            "evaluate_during_training": True,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "use_multiprocessing_for_evaluation": False,
            "use_multiprocessing": False,
            "dataloader_num_workers": 0,
            "save_strategy": "epoch",  # "steps"
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            # Training Duration and Early Stopping
            "num_train_epochs": self.epochs,
            "use_early_stopping": True,
            "early_stopping_patience": 3,
            "early_stopping_metric": "macro_f1",
            "early_stopping_metric_minimize": False,
            # Logging & WandB
            "logging_dir": os.path.join(self.output_dir, "logs"),
            "logging_strategy": "steps",
            "logging_steps": 1,
            "use_wandb": self.report_to == "wandb",
            "wandb_project": "tiald-joint-labels",
            "wandb_kwargs": {"job_type": "training", "name": f"{self.output_model_name}-{self.timestamp}"},
        }
        if self.model is None:
            self.model = TiALDMultiTaskModel(
                self.model_type,
                self.input_model_name,
                num_labels=self.num_labels,
                use_cuda=self.use_cuda,
                args=model_args,
                tiald_curriculum_loss=self.curriculum_loss,
            )
            print("Model args:", self.model.args)

        print(f"Starting {self.output_model_name} training...")
        self.model.tiald_total_training_steps = (len(self.train_df) // self.batch_size) * self.epochs
        print(f"  Total training steps: {self.model.tiald_total_training_steps}")
        self.model.tiald_global_step = 0
        self.model.train_model(
            train_df=self.train_df,
            eval_df=self.eval_df,
            accuracy=lambda y_true, y_pred: compute_task_metrics(
                y_true=y_true,
                y_pred=(y_pred > 0.5).astype(int),
            )["accuracy"],
            macro_f1=lambda y_true, y_pred: compute_task_metrics(
                y_true=y_true,
                y_pred=(y_pred > 0.5).astype(int),
            )["macro_f1"],
            weighted_f1=lambda y_true, y_pred: compute_task_metrics(
                y_true=y_true,
                y_pred=(y_pred > 0.5).astype(int),
            )["weighted_f1"],
        )
        print(f"Training completed for {self.output_model_name}")

        print(f"Evaluating {self.output_model_name} on validation set...")
        result, eval_model_outputs, wrong_predictions = self.model.eval_model(self.eval_df)
        print(f"Evaluation results for {self.output_model_name}:", result)

        y_pred_types, y_pred_sentiments, y_pred_topics = self.get_true_labels(eval_model_outputs)
        metrics = self._calculate_metrics(
            self.eval_df,
            y_pred_types=y_pred_types,
            y_pred_sentiments=y_pred_sentiments,
            y_pred_topics=y_pred_topics,
        )
        results = {
            "config": {
                "task": "tiald-joint-labels",
                "model_name": self.output_model_name,
                "max_seq_length": self.max_length,
                "train_batch_size": self.batch_size,
                "learning_rate": self.lr,
                "num_train_epochs": self.epochs,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "test_size": len(self.eval_df),
                "test_date": self.timestamp,
            },
            "abusiveness_metrics": metrics["abusiveness"],
            "sentiment_metrics": metrics["sentiment"],
            "topic_metrics": metrics["topic"],
        }
        metrics_file = os.path.join(self.output_dir, "eval_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)

    def load_model_for_inference(self, model_path=None):
        """Load a trained model from disk"""
        path_to_load = model_path if model_path else self.input_model_name

        print(f"Loading model for inference from {path_to_load}")
        self.model = TiALDMultiTaskModel(
            model_type=self.model_type,
            model_name=path_to_load,
            num_labels=self.num_labels,
            use_cuda=self.use_cuda,
        )
        return self.model

    def evaluate(self, test_df=None):
        """Make predictions and evaluate on new or test data"""
        if self.model is None:
            self.load_model_for_inference()

        if test_df is None:
            test_df = self.test_df

        texts = test_df.comment_clean.tolist()
        cids = test_df.comment_id.tolist()

        preds, model_outputs = self.model.predict(texts)
        y_pred_types, y_pred_sentiments, y_pred_topics = self.get_true_labels(model_outputs)
        metrics = self._calculate_metrics(
            test_df, y_pred_types=y_pred_types, y_pred_sentiments=y_pred_sentiments, y_pred_topics=y_pred_topics
        )

        results = {
            "config": {
                "task": "tiald-joint-labels",
                "model_name": self.input_model_name,
                "max_seq_length": self.max_length,
                "train_batch_size": self.batch_size,
                "learning_rate": self.lr,
                "num_train_epochs": self.epochs,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "test_size": len(test_df),
                "test_date": self.timestamp,
            },
            "abusiveness_metrics": metrics["abusiveness"],
            "sentiment_metrics": metrics["sentiment"],
            "topic_metrics": metrics["topic"],
            "abusiveness_predictions": {cid: pred for cid, pred in zip(cids, y_pred_types)},
            "sentiment_predictions": {cid: pred for cid, pred in zip(cids, y_pred_sentiments)},
            "topic_predictions": {cid: pred for cid, pred in zip(cids, y_pred_topics)},
        }
        pred_file = os.path.join(self.output_dir, "predictions_test.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to {pred_file}")
        return metrics

    def _calculate_metrics(
        self, df: pd.DataFrame, y_pred_types: list[str], y_pred_topics: list[str], y_pred_sentiments: list[str]
    ):
        """Calculate metrics for each task"""
        metrics = {
            "abusiveness": compute_task_metrics(y_true=df.abusiveness.tolist(), y_pred=y_pred_types),
            "sentiment": compute_task_metrics(y_true=df.sentiment.tolist(), y_pred=y_pred_sentiments),
            "topic": compute_task_metrics(y_true=df.topic.tolist(), y_pred=y_pred_topics),
        }
        return metrics
