"""
Shared evaluation plots for all models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _plot(matrix, labels, fmt, title, path, **heatmap_kwargs):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels, **heatmap_kwargs)
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.show()


def save_confusion_matrices(y_true, y_pred, labels, results_dir: Path, model_name: str) -> np.ndarray:
    """Save count and row-normalized confusion matrices to results_dir; return the counts."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    assert cm.sum() == len(y_true)
    # Row-normalized: share of each true emotion
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _plot(cm, labels, 'd', f'{model_name} Confusion Matrix (Counts)',
          results_dir / 'confusion_matrix.png')
    _plot(cm_normalized, labels, '.2f', f'{model_name} Confusion Matrix (Normalized)',
          results_dir / 'confusion_matrix_normalized.png', vmin=0.0, vmax=1.0)
    return cm
