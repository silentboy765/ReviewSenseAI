"""Ground-truth-only model evaluation helpers."""

import re

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


def normalize_label(value):
    if pd.isna(value):
        return None
    clean = re.sub(r"[^a-z]+", " ", str(value).strip().lower()).strip()
    return {
        "positive": "Positive",
        "pos": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
    }.get(clean)


def evaluate_predictions(actual_values, predicted_values):
    """Compute metrics only for recognized human-provided labels."""
    valid_pairs = [
        (normalize_label(actual), normalize_label(predicted))
        for actual, predicted in zip(actual_values, predicted_values)
    ]
    valid_pairs = [(actual, predicted) for actual, predicted in valid_pairs if actual and predicted]
    if not valid_pairs:
        raise ValueError("No valid ground-truth sentiment labels were found.")
    actual, predicted = zip(*valid_pairs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=SENTIMENT_LABELS,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=SENTIMENT_LABELS),
        "labels": SENTIMENT_LABELS,
        "evaluated_rows": len(actual),
    }
