"""
preprocessing.py
-----------------
Text cleaning utilities for the Movie Review Analysis System.

Kept intentionally simple so the whole pipeline is easy to explain
in a college viva: lowercase -> strip HTML/punctuation/noise ->
collapse whitespace -> (optional) drop stopwords.
"""

import re

# A small, hand-picked stopword list so we don't need to pull in nltk
# just for this. Kept short on purpose -- we deliberately keep words
# like "not", "no", "very" because they carry sentiment information
# and stripping them tends to hurt a sentiment classifier.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "as", "by", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "their", "our", "your", "my", "me", "us", "him", "which", "who",
    "whom", "what", "there", "here", "am", "will", "would", "shall",
    "should", "can", "could", "do", "does", "did", "have", "has", "had",
}

HTML_TAG_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean a single piece of review text so it is ready for TF-IDF.

    Steps:
      1. Handle missing / non-string input safely.
      2. Lowercase everything.
      3. Strip HTML tags (the IMDB dataset is full of <br /> tags).
      4. Strip URLs.
      5. Remove punctuation & special characters (keep only letters).
      6. Collapse repeated whitespace.
      7. Optionally drop common stopwords (kept off by default because
         short reviews lose useful signal quickly).

    Returns an empty string for missing/empty input rather than raising,
    so callers can handle "empty after cleaning" as a validation case.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        words = [w for w in text.split() if w not in STOPWORDS]
        text = " ".join(words)

    return text


def is_valid_review(raw_text: str) -> bool:
    """A review is valid if it still has content after cleaning."""
    if raw_text is None:
        return False
    if not raw_text.strip():
        return False
    return len(preprocess_text(raw_text)) > 0
