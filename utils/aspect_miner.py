"""Lightweight, non-generative phrase-frequency analysis."""

from sklearn.feature_extraction.text import CountVectorizer


def extract_ngrams(texts: list[str], limit: int = 20) -> list[tuple[str, int]]:
    """Return frequent bigrams and trigrams; these counts are not generated explanations."""
    cleaned = [text.strip() for text in texts if text and text.strip()]
    if not cleaned:
        return []
    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        stop_words="english",
        max_features=4000,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return []
    frequencies = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, frequencies), key=lambda item: (-int(item[1]), item[0]))
    return [(term, int(count)) for term, count in ranked[:limit]]
