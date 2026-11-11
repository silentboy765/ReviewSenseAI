"""VADER fallback and comparison predictions."""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_vader(texts: list[str]) -> list[dict[str, float | str]]:
    """Return VADER labels and compound scores for a list of review strings."""
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        compound = analyzer.polarity_scores(text)["compound"] if text.strip() else 0.0
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        results.append({"sentiment": sentiment, "compound": float(compound)})
    return results
