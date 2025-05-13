import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Valid labels for each task
TASK_INFO = {
    "abusiveness": {
        "label_list": ("Abusive", "Not Abusive"),
        "label_column": "abusiveness",
        "description": "Abusiveness Detection",
    },
    "topic": {
        "label_list": ("Political", "Racial", "Religious", "Sexist", "Other"),
        "label_column": "topic",
        "description": "Topic Classification",
    },
    "sentiment": {
        "label_list": ("Positive", "Neutral", "Negative", "Mixed"),
        "label_column": "sentiment",
        "description": "Sentiment Analysis",
    },
}

MULTITASK_LOSS_AGG = ["avg", "sum"]


def get_label_maps(task: str) -> tuple[dict[str, int], dict[int, str], list[str]]:
    """Return label2id, id2label mappings and list of labels for the task."""
    label_list = TASK_INFO[task]["label_list"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label, label_list


def compute_task_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute accuracy, macro F1, precision, and recall."""
    return {
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        "macro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "macro_recall": recall_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "macro_precision": precision_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "weighted_f1": f1_score(y_true=y_true, y_pred=y_pred, average="weighted"),
        "weighted_recall": recall_score(y_true=y_true, y_pred=y_pred, average="weighted"),
        "weighted_precision": precision_score(y_true=y_true, y_pred=y_pred, average="weighted"),
        "per_class_metrics": classification_report(y_true=y_true, y_pred=y_pred, zero_division=0, output_dict=True),
    }
