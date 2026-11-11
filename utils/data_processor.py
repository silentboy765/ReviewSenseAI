"""CSV ingestion and safe analysis output construction."""

import json
import re

import pandas as pd

from utils.vader_analyzer import analyze_vader

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_REVIEWS_PER_UPLOAD = 100


def normalize_column_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def suggest_review_column(dataframe: pd.DataFrame):
    if dataframe.shape[1] == 0:
        raise ValueError("The dataset does not contain any columns.")
    normalized = {normalize_column_name(column): column for column in dataframe.columns}
    for candidate in ("review", "reviews", "review text", "comment", "feedback", "text", "content"):
        if candidate in normalized:
            return normalized[candidate]
    text_columns = [
        column for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    return text_columns[0] if text_columns else dataframe.columns[0]


def detect_column(dataframe: pd.DataFrame, candidate_names: tuple[str, ...]):
    normalized = {normalize_column_name(column): column for column in dataframe.columns}
    return next((normalized[name] for name in candidate_names if name in normalized), None)


def detect_rating_column(dataframe: pd.DataFrame):
    return detect_column(dataframe, ("rating", "ratings", "stars", "star rating", "product rating"))


def detect_label_column(dataframe: pd.DataFrame):
    return detect_column(dataframe, ("sentiment", "label", "sentiment label", "ground truth", "ground truth sentiment"))


def load_reviews_csv(file_object) -> pd.DataFrame:
    uploaded_size = getattr(file_object, "size", None)
    if uploaded_size is not None and uploaded_size > MAX_UPLOAD_BYTES:
        raise ValueError("This CSV is over the 2 MB upload limit. Split it into smaller files and try again.")
    try:
        file_object.seek(0)
        dataframe = pd.read_csv(file_object)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The uploaded CSV is empty. Add a header and at least one review.") from error
    except pd.errors.ParserError as error:
        raise ValueError("The CSV could not be parsed. Check its quoting and column separators.") from error
    except UnicodeDecodeError as error:
        raise ValueError("Could not decode the CSV. Save it as UTF-8 and upload it again.") from error
    if dataframe.empty:
        raise ValueError("The CSV has headers but no data rows.")
    if dataframe.shape[1] == 0:
        raise ValueError("The CSV does not contain any columns.")
    if len(dataframe) > MAX_REVIEWS_PER_UPLOAD:
        raise ValueError(
            f"This app accepts at most {MAX_REVIEWS_PER_UPLOAD} reviews per upload to protect the free AI quota. "
            "Split the CSV into smaller files and try again."
        )
    return dataframe


def prepare_texts(dataframe: pd.DataFrame, review_column) -> list[str]:
    if review_column not in dataframe.columns:
        raise ValueError(f"Review column '{review_column}' was not found.")
    return dataframe[review_column].fillna("").astype(str).str.strip().tolist()


def choose_output_names(columns) -> dict[str, str]:
    """Generate analysis field names that never overwrite original CSV columns."""
    used = set(columns)
    result = {}
    for key, preferred in (
        ("ai", "AI Sentiment"),
        ("confidence", "AI Confidence"),
        ("vader", "VADER Sentiment"),
        ("vader_score", "VADER Compound"),
        ("agreement", "Methods Agree"),
        ("duplicate", "Duplicate Review"),
        ("vader_fallback", "VADER fallback sentiment"),
        ("reason", "AI Explanation"),
        ("aspects", "Detected Aspects"),
        ("source", "Prediction Source"),
    ):
        name, suffix = preferred, 2
        while name in used:
            name = f"{preferred} ({suffix})"
            suffix += 1
        result[key] = name
        used.add(name)
    return result


def add_vader_columns(dataframe: pd.DataFrame, texts: list[str], names: dict) -> pd.DataFrame:
    output = dataframe.copy()
    vader_results = analyze_vader(texts)
    output[names["vader"]] = [item["sentiment"] for item in vader_results]
    output[names["vader_score"]] = [item["compound"] for item in vader_results]
    review_values = pd.Series(texts, index=output.index)
    output[names["duplicate"]] = review_values.ne("") & review_values.duplicated(keep=False)
    return output


def add_prediction_columns(dataframe: pd.DataFrame, predictions: list[dict], names: dict) -> pd.DataFrame:
    output = dataframe.copy()
    output[names["ai"]] = [item["sentiment"] for item in predictions]
    output[names["confidence"]] = [item.get("confidence") for item in predictions]
    output[names["agreement"]] = output[names["ai"]] == output[names["vader"]]
    for key, getter in (
        ("reason", lambda item: item.get("reason", "")),
        (
            "aspects",
            lambda item: json.dumps(item.get("aspects", []), ensure_ascii=False),
        ),
        ("source", lambda item: item.get("source", "Cloudflare Workers AI")),
    ):
        if key in names:
            output[names[key]] = [getter(item) for item in predictions]
    return output


def top_keywords(review_series, limit: int = 10) -> list[tuple[str, int]]:
    """Count simple words; use aspect_miner.extract_ngrams for multi-word phrases."""
    stop_words = {
        "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "but", "by", "can", "could", "did", "do",
        "does", "during", "each", "even", "for", "from", "get", "got", "had", "has", "have",
        "he", "her", "here", "him", "his", "i", "if", "in", "into", "is", "it", "its", "just",
        "me", "more", "most", "my", "no", "not", "of", "on", "one", "only", "or", "our", "out",
        "over", "really", "she", "so", "some", "than", "that", "the", "their", "them", "there",
        "they", "this", "those", "through", "to", "too", "up", "us", "very", "was", "we", "were",
        "what", "when", "which", "who", "will", "with", "would", "you", "your",
    }
    words = []
    for review in review_series.fillna("").astype(str):
        words.extend(
            word for word in re.findall(r"[a-zA-Z][a-zA-Z'-]{1,}", review.lower())
            if word not in stop_words
        )
    from collections import Counter

    return Counter(words).most_common(limit)


def keyword_comparison(dataframe: pd.DataFrame, review_column, label_column, target: str):
    reviews = dataframe.loc[dataframe[label_column] == target, review_column]
    return top_keywords(reviews)
