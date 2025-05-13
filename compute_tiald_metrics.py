#!/usr/bin/env python3
"""
Evaluation Script for TiALD Benchmark

This script evaluates model predictions against the TiALD test set and computes
performance metrics for abusiveness, topic, and sentiment classification tasks.

Usage:
    python compute_tiald_metrics.py -p <model-predictions.json> [-o <output-results.json>]
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict

from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report

random.seed(42)

TiALD_TASKs = {
    "abusiveness": {
        "label_list": ("Abusive", "Not Abusive"),
        "description": "Abusiveness Detection",
    },
    "sentiment": {
        "label_list": ("Positive", "Neutral", "Negative", "Mixed"),
        "description": "Sentiment Analysis",
    },
    "topic": {
        "label_list": ("Political", "Racial", "Sexist", "Religious", "Other"),
        "description": "Topic Classification",
    },
}


def load_predictions(prediction_file: str) -> dict[str, dict[str, str]]:
    """Load model predictions from a JSON file."""
    with open(prediction_file, "r", encoding="utf-8") as fin:
        predictions_dict = json.load(fin)

    required_keys = [f"{task}_predictions" for task in TiALD_TASKs.keys()]
    for key in required_keys:
        if key not in predictions_dict:
            raise ValueError(f"Missing required key '{key}' in prediction file!")

    return predictions_dict


def compute_task_metrics(y_true: list, y_pred: list, labels: list = None) -> dict[str, float]:
    """
    Generalize classification report function that handles invalid predictions
    (predictions not found in the true label set).

    Parameters:
    - y_true: Array of true labels
    - y_pred: Array of predicted labels (may contain invalid labels)
    - labels: List of labels (default: None)

    Returns:
    - Dictionary with metrics:
        - accuracy
        - macro (F1, recall, precision),
        - weighted (F1, recall, precision),
        - per-class metrics,
        - invalid count, rate, and total samples.
    """

    if not y_true:
        print("Warning: No ground truth labels provided!")
        return {}
    if not y_pred:
        print("Warning: No predicted labels provided!")
        return {}

    # Handle invalid predictions
    _valid_classes = set(y_true)
    _pred_classes = set(y_pred)
    invalid_count = 0
    if _pred_classes.difference(_valid_classes):
        _y_preds = []
        for _true, _pred in zip(y_true, y_pred):
            if _pred not in _valid_classes:
                invalid_count += 1
                _y_preds.append(random.choice(list(_valid_classes.difference({_true}))))
            else:
                _y_preds.append(_pred)
        y_pred = _y_preds

    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    assert set(y_pred).issubset(_valid_classes), "y_pred must contain all valid classes"

    _metrics = classification_report(y_true=y_true, y_pred=y_pred, labels=labels, zero_division=0, output_dict=True)
    return {
        "accuracy": _metrics["accuracy"],
        "macro_f1": _metrics["macro avg"]["f1-score"],
        "macro_recall": _metrics["macro avg"]["recall"],
        "macro_precision": _metrics["macro avg"]["precision"],
        "weighted_f1": _metrics["weighted avg"]["f1-score"],
        "weighted_recall": _metrics["weighted avg"]["recall"],
        "weighted_precision": _metrics["weighted avg"]["precision"],
        "per_class_metrics": _metrics,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(y_pred),
        "total_samples": len(y_true),
    }


def compute_tiald_metrics(test_dataset: Dataset, model_preds_dict: Dict) -> Dict:
    """
    Compute TiALD metrics from a prediction file and save results to a file.

    Args:
        test_dataset (Dataset): The test dataset.
        model_preds_dict (Dict): The model predictions.

    Returns:
        Dict: The computed metrics.
    """

    config = model_preds_dict.get("config", {})
    config["evaluation_date"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"config": config}

    missing_predictions = defaultdict(list)
    invalid_predictions = defaultdict(list)
    for task in TiALD_TASKs.keys():
        _cid_to_preds = model_preds_dict.get(f"{task}_predictions", {})
        if not _cid_to_preds:
            continue
        _task_labels = TiALD_TASKs[task]["label_list"]
        y_true, y_pred = [], []
        for sample in test_dataset:
            cid = str(sample["comment_id"])
            y_true.append(sample[task])
            if cid not in _cid_to_preds:
                missing_predictions[task].append(cid)
                y_pred.append("Missing")
            elif _cid_to_preds[cid] not in _task_labels:
                invalid_predictions[task].append(cid)
                y_pred.append("Invalid")
            else:
                y_pred.append(_cid_to_preds[cid])
        results[f"{task}_metrics"] = compute_task_metrics(y_true=y_true, y_pred=y_pred, labels=_task_labels)

    results["missing_predictions"] = dict(missing_predictions)
    results["invalid_predictions"] = dict(invalid_predictions)
    return results


def evaluate_predictions(
    prediction_file: str, output_result_file: str = None, append_metrics: bool = False, quiet: bool = False
) -> None:
    """
    Evaluate model predictions against the TiALD test set and save results to a file.

    Args:
        prediction_file (str): Path to the prediction JSON file.
        output_result_file (str, optional): Path to save the evaluation results.
        append_metrics (bool, optional): Append metrics to the predictions file.
        quiet (bool, optional): Skip printing the results to the console.

    Returns:
        None
    """

    try:
        print("Loading TiALD test dataset...")
        tiald_dataset = load_dataset("fgaim/tigrinya-abusive-language-detection", split="test")
    except FileNotFoundError:
        print("Error: TiALD dataset not found. Please check the path.")
        return

    print(f"Evaluating predictions from {prediction_file}...")
    model_predictions_dict = load_predictions(prediction_file)
    tiald_metrics = compute_tiald_metrics(tiald_dataset, model_predictions_dict)

    # Append metrics to the predictions file if specified
    if append_metrics:
        model_predictions_dict.update(tiald_metrics)
        with open(prediction_file, "w", encoding="utf-8") as fout:
            json.dump(model_predictions_dict, fout, indent=2, ensure_ascii=False)
        print(f"\nMetrics appended to {prediction_file}")

    if output_result_file:
        with open(output_result_file, "w", encoding="utf-8") as fout:
            json.dump(tiald_metrics, fout, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_result_file}")

    if quiet:
        return

    print("\n===== EVALUATION RESULTS =====")
    print(f"Model: {tiald_metrics['config'].get('model_name', 'Unnamed')}")
    print(f"Evaluation Date: {tiald_metrics['config']['evaluation_date']}")
    print(f"\nMissing Predictions: {tiald_metrics['missing_predictions']}")
    print(f"Invalid Predictions: {tiald_metrics['invalid_predictions']}")

    for task in TiALD_TASKs.keys():
        task_metrics = tiald_metrics.get(f"{task}_metrics")
        if not task_metrics:
            print(f"\nNo metrics found for the {task} task!")
            continue
        print(f"\n{task.capitalize()} metrics:")
        print(f"- Macro F1: {task_metrics['macro_f1']:.4f}")
        print(f"- Accuracy: {task_metrics['accuracy']:.4f}")
        print("- Per-class F1 scores:")
        for label in TiALD_TASKs[task]["label_list"]:
            print(f"    - {label}: {task_metrics['per_class_metrics'][label]['f1-score']:.4f}")


def parse_cli_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model predictions against the TiALD test set")
    parser.add_argument("-p", "--prediction_file", nargs="+", required=True, help="Path to the predictions JSON file")
    parser.add_argument("-o", "--output_file", nargs="+", default=None, help="Path to save the metrics JSON file")
    parser.add_argument("-a", "--append_metrics", action="store_true", help="Append metrics to the predictions file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Skip printing the results to the console")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    prediction_files = args.prediction_file
    if args.output_file is not None:
        if len(args.output_file) != len(prediction_files):
            raise ValueError("Number of output files must match the number of prediction files!")
    else:
        output_files = [None] * len(prediction_files)

    for prediction_file, output_file in zip(prediction_files, output_files):
        evaluate_predictions(
            prediction_file=prediction_file,
            output_result_file=output_file,
            append_metrics=args.append_metrics,
            quiet=args.quiet,
        )
