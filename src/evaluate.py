"""Evaluation utilities and plots."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src import config


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "roc_auc": metrics.roc_auc_score(y_true, y_proba),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str) -> None:
    """Save precision-recall curve plot."""
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def permutation_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str,
) -> pd.DataFrame:
    """Compute and save permutation importance plot."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    top = importances.head(15)
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return importances


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        "rmse": float(metrics.mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }
