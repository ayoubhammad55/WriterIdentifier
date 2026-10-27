"""
Evaluation utilities shared by all model pipelines.

Functions
---------
save_metrics     – persist a metrics dict as JSON
plot_history     – plot train/val loss and accuracy curves
plot_confusion   – plot a labelled confusion matrix heatmap
compare_models   – build a summary comparison DataFrame
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for Kaggle)
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import pandas as pd
    _PD_OK = True
except ImportError:
    _PD_OK = False

from utils.constants import RESULTS_DIR


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _jsonable(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_metrics(metrics: dict, name: str, out_dir: str = RESULTS_DIR) -> str:
    """
    Save a metrics dict as a JSON file under ``out_dir/<name>_metrics.json``.

    Skips non-serialisable keys (e.g. 'confusion_matrix' if it's very large;
    it is still included as a nested list).

    Returns the output file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_metrics.json")

    safe = _jsonable(metrics)
    # Drop the human-readable report string (already printed to stdout)
    safe.pop("report", None)

    with open(path, "w") as f:
        json.dump(safe, f, indent=2)

    print(f"[Eval] Metrics saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_history(
    history: dict,
    title:   str = "Training history",
    out_dir: str = RESULTS_DIR,
    name:    str = "history",
) -> Optional[str]:
    """
    Plot loss and accuracy curves from a training history dict.

    Expected keys: train_loss, train_acc, val_loss, val_acc.
    Returns path to saved figure, or None if matplotlib is unavailable.
    """
    if not _MPL_OK:
        print("[Eval] matplotlib not available — skipping plot.")
        return None

    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].plot(epochs, history.get("train_loss", []), label="Train")
    axes[0].plot(epochs, history.get("val_loss",   []), label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.get("train_acc", []), label="Train")
    axes[1].plot(epochs, history.get("val_acc",   []), label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Training history plot -> {path}")
    return path


def plot_confusion(
    cm:          np.ndarray,
    title:       str  = "Confusion matrix",
    out_dir:     str  = RESULTS_DIR,
    name:        str  = "confusion",
    max_classes: int  = 40,
) -> Optional[str]:
    """
    Plot a confusion matrix heatmap.  If there are more than ``max_classes``
    classes, only the diagonal (accuracy per class) is plotted instead.

    Returns path to saved figure, or None.
    """
    if not _MPL_OK:
        return None

    os.makedirs(out_dir, exist_ok=True)
    n = cm.shape[0]

    fig, ax = plt.subplots(figsize=(min(n, max_classes) * 0.3 + 4,
                                    min(n, max_classes) * 0.3 + 4))

    if n <= max_classes:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    else:
        # Show per-class accuracy only
        per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-9)
        ax.bar(range(n), per_class, width=0.8)
        ax.set_title(f"{title} — per-class accuracy ({n} classes)")
        ax.set_xlabel("Class index")
        ax.set_ylabel("Accuracy")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Confusion matrix plot -> {path}")
    return path


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def compare_models(results: Dict[str, dict]) -> Optional[object]:
    """
    Build a comparison table from a dict of {model_name: metrics_dict}.

    Returns a pandas DataFrame if pandas is available, else prints a summary.

    Example
    -------
    compare_models({
        "LBP+SVM":  {"accuracy": 0.95, "macro_f1": 0.94, ...},
        "CNN":      {"accuracy": 0.91, "macro_f1": 0.90, ...},
        "ResNet18": {"accuracy": 0.93, "macro_f1": 0.92, ...},
    })
    """
    rows = []
    for model_name, m in results.items():
        rows.append(
            {
                "Model":                  model_name,
                "Test Accuracy":          f"{m.get('accuracy', float('nan')):.4f}",
                "Macro F1":               f"{m.get('macro_f1', float('nan')):.4f}",
                "Inference (ms/img)":     f"{m.get('inference_time_per_image_s', float('nan')) * 1000:.2f}",
                "Train Time (s)":         f"{m.get('total_time_s', m.get('train_time_', float('nan'))):.1f}",
            }
        )

    if _PD_OK:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60 + "\n")
        return df

    # Fallback: plain text
    print("\n=== Model Comparison ===")
    for r in rows:
        print("  " + " | ".join(f"{k}: {v}" for k, v in r.items()))
    print()
    return rows
