from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def classify_stable(
    truth: Iterable[float],
    pred: Iterable[float],
    threshold: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    each_true = pd.Series(truth)
    each_pred = pd.Series(pred)
    actual_pos = each_true <= threshold
    actual_neg = each_true > threshold
    model_pos = each_pred <= threshold
    model_neg = each_pred > threshold
    nan_mask = each_pred.isna()
    model_pos[nan_mask] = False
    model_neg[nan_mask] = True
    return (
        actual_pos & model_pos,
        actual_pos & model_neg,
        actual_neg & model_pos,
        actual_neg & model_neg,
    )


def stable_metrics(truth: Iterable[float], pred: Iterable[float], threshold: float = 0.0) -> dict[str, float]:
    tp, fn, fp, tn = map(sum, classify_stable(truth, pred, threshold))
    total_pos = tp + fn
    total_neg = tn + fp
    prevalence = total_pos / (total_pos + total_neg) if (total_pos + total_neg) else math.nan
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / total_pos if total_pos else math.nan
    fpr = fp / total_neg if total_neg else math.nan
    fnr = fn / total_pos if total_pos else math.nan
    accuracy = (tp + tn) / (total_pos + total_neg) if (total_pos + total_neg) else math.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else math.nan
    daf = precision / prevalence if prevalence and not math.isnan(prevalence) else math.nan

    each_true = pd.Series(truth, dtype=float)
    each_pred = pd.Series(pred, dtype=float)
    valid = ~(each_true.isna() | each_pred.isna())
    errors = each_pred[valid] - each_true[valid]

    return {
        "F1": f1,
        "DAF": daf,
        "FPR": fpr,
        "FNR": fnr,
        "Accuracy": accuracy,
        "MAE": errors.abs().mean(),
        "ME": errors.mean(),
        "RMSE": np.sqrt(np.mean(np.square(errors))) if len(errors) else math.nan,
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
    }


def gini(values: Iterable[float]) -> float:
    array = np.array(list(values), dtype=float)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return math.nan
    array = np.sort(array)
    n = len(array)
    cumulative = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n if cumulative[-1] else 0.0


def performance_disparity_ratio(values: Iterable[float], quantile: float = 0.1) -> float:
    series = pd.Series(list(values), dtype=float).dropna().sort_values()
    if series.empty:
        return math.nan
    k = max(1, int(len(series) * quantile))
    best = series.iloc[:k].mean()
    worst = series.iloc[-k:].mean()
    return worst / best if best else math.nan
