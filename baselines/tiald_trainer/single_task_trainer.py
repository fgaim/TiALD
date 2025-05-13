"""
Single-Task Training for TiALD
"""

import json
import os
from datetime import datetime
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .utils import TASK_INFO, compute_task_metrics, get_label_maps

set_seed(42)


class TiALDSingleTaskTrainer:
    def __init__(
        self,
        task: str,
        input_model_name: str,
        output_model_name: str,
        dataset_name: str,
        hf_token: str = None,
        output_dir: str = "./results",
        batch_size: int = 8,
        lr: float = 2e-5,
        epochs: int = 4,
        max_length: int = 256,
        report_to: str = "none",
    ):
        self.task = task
        self.input_model_name = input_model_name
        self.output_model_name = (
            output_model_name
            if output_model_name.lower().endswith(f"{task.lower()}")
            else f"{output_model_name}_{task}"
        )
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.output_model_path = os.path.join(output_dir, self.output_model_name)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.max_length = max_length
        self.report_to = report_to
        self.hf_token = hf_token

        self.label2id, self.id2label, self.label_list = get_label_maps(task)
        self.num_labels = len(self.label_list)

        self.tokenizer = None
        self.data_collator = None
        self.dataset = None

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_data_and_model()

    def _setup_data_and_model(self):
        """Setup reporting and load the dataset"""
        if self.report_to == "wandb":
            wandb.init(
                project="tiald-single-task",
                name=f"{self.output_model_name}-{self.timestamp}",
                config={
                    "task": self.task,
                    "input_model": self.input_model_name,
                    "output_model": self.output_model_name,
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                },
            )
        os.makedirs(self.output_model_path, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.input_model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.dataset = self._load_and_process_dataset()

    def _tokenize_comment(self, example: dict[str, Any]) -> dict[str, Any]:
        """Tokenize a batch of examples using the provided tokenizer."""
        return self.tokenizer(
            example["comment_clean"], truncation=True, padding="max_length", max_length=self.max_length
        )

    def _load_and_process_dataset(self):
        """
        Load and process the TiALD dataset for multi-task learning.

        Returns:
            Dictionary containing train, validation, and test datasets
        """
        print(f"Loading dataset {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, token=self.hf_token)
        print("Tokenizing dataset...")
        tokenized = dataset.map(self._tokenize_comment, batched=True)
        print("Processing task labels...")
        label_column = TASK_INFO[self.task]["label_column"]
        return tokenized.map(lambda example: {"label": self.label2id[example[label_column]]})

    def train(self, learning_rate=None, batch_size=None, epochs=None):
        print(f"Loading model from {self.input_model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(self.input_model_name, num_labels=self.num_labels)

        training_args = TrainingArguments(
            output_dir=self.output_model_path,
            run_name=f"train-{self.output_model_name}-{self.timestamp}",
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            learning_rate=learning_rate if learning_rate else self.lr,
            per_device_train_batch_size=batch_size if batch_size else self.batch_size,
            per_device_eval_batch_size=batch_size if batch_size else self.batch_size * 2,
            num_train_epochs=epochs if epochs else self.epochs,
            warmup_ratio=0.1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            report_to=self.report_to,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Training model...")
        trainer.train()
        print("Evaluating model (default on test set)...")
        metrics = trainer.evaluate()
        print("Validation metrics:", metrics)
        print("Saving model...")
        trainer.save_model(self.output_model_path)

        results = {
            "config": {
                "task": self.task,
                "model_name": self.output_model_name,
                "max_seq_length": self.max_length,
                "train_batch_size": self.batch_size,
                "learning_rate": self.lr,
                "num_train_epochs": self.epochs,
                "test_size": len(self.dataset["validation"]),
                "test_date": self.timestamp,
            },
            f"{self.task}_metrics": metrics,
        }
        with open(f"{self.output_model_path}/eval_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"{self.output_model_path}/training_args.json", "w") as f:
            json.dump(training_args.to_dict(), f, indent=2)

        self.trainer = trainer
        self.model = model

    def evaluate(self, test_split="test"):
        if self.input_model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.input_model_name)
        elif not hasattr(self, "model"):
            raise ValueError("Model not trained or loaded from checkpoint.")

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            args=TrainingArguments(
                run_name=f"eval-{self.input_model_name}-{self.timestamp}",
                per_device_eval_batch_size=self.batch_size,
                do_train=False,
                report_to=self.report_to,
            ),
        )

        test_set = self.dataset[test_split]
        test_results = trainer.predict(test_set)
        pred_ids = np.argmax(test_results.predictions, axis=-1)

        results = {
            "config": {
                "task": self.task,
                "model_name": self.input_model_name,
                "test_size": len(test_set),
                "test_date": self.timestamp,
            },
            f"{self.task}_metrics": test_results.metrics,
            "abusiveness_predictions": {},
            "topic_predictions": {},
            "sentiment_predictions": {},
        }

        for i, ex in enumerate(test_set):
            cid = ex["comment_id"]
            pred = self.id2label[pred_ids[i]]
            if self.task == "abusiveness":
                results["abusiveness_predictions"][cid] = pred
            elif self.task == "topic":
                results["topic_predictions"][cid] = pred
            elif self.task == "sentiment":
                results["sentiment_predictions"][cid] = pred

        pred_file = os.path.join(self.output_model_path, f"predictions_{test_split}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to {pred_file}")
        return test_results.metrics

    def _compute_metrics(self, eval_pred) -> dict[str, float]:
        """Compute accuracy, macro F1, precision, and recall."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # per_class_f1 = f1_score(labels, preds, average=None)
        return compute_task_metrics(y_true=labels, y_pred=preds)
