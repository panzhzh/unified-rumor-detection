"""Evaluation metrics for rumor detection"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from typing import Dict, List, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: List[int] = [0, 1]
) -> Dict[str, float]:
    """Compute comprehensive metrics for rumor detection

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        labels: List of label values

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (macro and weighted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    metrics['precision_macro'] = precision
    metrics['recall_macro'] = recall
    metrics['f1_macro'] = f1

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    for i, label in enumerate(labels):
        label_name = 'real' if label == 0 else ('fake' if label == 1 else 'unverified')
        metrics[f'precision_{label_name}'] = precision_per_class[i]
        metrics[f'recall_{label_name}'] = recall_per_class[i]
        metrics[f'f1_{label_name}'] = f1_per_class[i]
        metrics[f'support_{label_name}'] = support[i]

    # AUC (for binary classification)
    if y_prob is not None and len(labels) == 2:
        try:
            if y_prob.ndim == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """Get detailed classification report

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Names of the classes

    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def compute_dataset_specific_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_names: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each dataset separately

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        dataset_names: Dataset names for each sample

    Returns:
        Dictionary mapping dataset names to their metrics
    """
    dataset_metrics = {}
    unique_datasets = np.unique(dataset_names)

    for dataset in unique_datasets:
        mask = dataset_names == dataset
        if np.sum(mask) > 0:
            dataset_metrics[dataset] = compute_metrics(
                y_true[mask],
                y_pred[mask]
            )

    return dataset_metrics