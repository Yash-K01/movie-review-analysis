"""
data_handler.py
-----------------
All CSV I/O for the Movie Review Analysis System lives here, so the
Streamlit pages stay focused on UI and utils/model.py stays focused
on ML. Nothing in this file touches Streamlit.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORICAL_PATH = os.path.join(DATA_DIR, "movie_reviews.csv")
USER_REVIEWS_PATH = os.path.join(DATA_DIR, "user_reviews.csv")

REQUIRED_COLUMNS = ["movie_name", "review", "sentiment"]


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def ensure_user_reviews_file():
    """Create user_reviews.csv with the right headers if it's missing."""
    ensure_data_dir()
    if not os.path.exists(USER_REVIEWS_PATH):
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(
            USER_REVIEWS_PATH, index=False
        )


def load_historical_data() -> pd.DataFrame:
    """
    Load the historical movie_reviews.csv dataset.

    Raises a clear, catchable error if the file is missing or malformed
    so the UI layer can show a friendly message instead of a traceback.
    """
    ensure_data_dir()
    if not os.path.exists(HISTORICAL_PATH):
        raise FileNotFoundError(
            f"Historical dataset not found at '{HISTORICAL_PATH}'. "
            "Please add a movie_reviews.csv file with columns: "
            "movie_name, review, sentiment."
        )

    df = pd.read_csv(HISTORICAL_PATH)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"movie_reviews.csv is missing required column(s): {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    if df.empty:
        raise ValueError("movie_reviews.csv was found but contains no rows.")

    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["movie_name"] = df["movie_name"].fillna("Unknown").astype(str).str.strip()

    if df.empty:
        raise ValueError(
            "movie_reviews.csv has no valid rows after cleaning "
            "(sentiment must be 'positive' or 'negative')."
        )

    if df["sentiment"].nunique() < 2:
        raise ValueError(
            "The dataset only contains one sentiment class. "
            "At least both 'positive' and 'negative' examples are required "
            "to train a classifier."
        )

    return df.reset_index(drop=True)


def load_user_reviews() -> pd.DataFrame:
    ensure_user_reviews_file()
    try:
        df = pd.read_csv(USER_REVIEWS_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    return df[REQUIRED_COLUMNS]


def save_user_review(movie_name: str, review: str, sentiment: str) -> None:
    """
    Append a single reviewed record to user_reviews.csv without
    overwriting existing rows. Safe to call on every "Analyze" click
    (each click is a distinct explicit user action, so no dedup logic
    is needed beyond that).
    """
    ensure_user_reviews_file()
    new_row = pd.DataFrame(
        [{
            "movie_name": movie_name.strip(),
            "review": review.strip(),
            "sentiment": sentiment.strip().lower(),
        }]
    )
    file_exists_with_header = (
        os.path.exists(USER_REVIEWS_PATH)
        and os.path.getsize(USER_REVIEWS_PATH) > 0
    )
    new_row.to_csv(
        USER_REVIEWS_PATH,
        mode="a",
        header=not file_exists_with_header,
        index=False,
    )


def get_combined_dataset() -> pd.DataFrame:
    """
    Historical data + user-submitted data, combined for future retraining.

    NOTE: user_reviews.csv stores the MODEL'S PREDICTED sentiment, not a
    human-verified ground-truth label. Combining it back into training
    data is therefore a convenience for experimentation, not a
    correctness guarantee -- predicted labels can reinforce the model's
    own mistakes. A production system should let a human confirm/correct
    each label before it's used for retraining.
    """
    historical = load_historical_data()
    user = load_user_reviews()
    user = user.dropna(subset=["review", "sentiment"])
    if user.empty:
        return historical
    user["sentiment"] = user["sentiment"].astype(str).str.strip().str.lower()
    user = user[user["sentiment"].isin(["positive", "negative"])]
    combined = pd.concat([historical, user], ignore_index=True)
    return combined


def get_movie_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-movie positive/negative percentage breakdown.

    Returns a DataFrame with columns:
        movie_name, total_reviews, positive_reviews, negative_reviews,
        positive_pct, negative_pct
    sorted by total_reviews descending.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "movie_name", "total_reviews", "positive_reviews",
                "negative_reviews", "positive_pct", "negative_pct",
            ]
        )

    grouped = (
        df.groupby("movie_name")["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for col in ["positive", "negative"]:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["total_reviews"] = grouped["positive"] + grouped["negative"]
    grouped["positive_pct"] = np.where(
        grouped["total_reviews"] > 0,
        (grouped["positive"] / grouped["total_reviews"] * 100).round(2),
        0.0,
    )
    grouped["negative_pct"] = np.where(
        grouped["total_reviews"] > 0,
        (grouped["negative"] / grouped["total_reviews"] * 100).round(2),
        0.0,
    )

    result = grouped.reset_index().rename(
        columns={"positive": "positive_reviews", "negative": "negative_reviews"}
    )
    result = result[[
        "movie_name", "total_reviews", "positive_reviews",
        "negative_reviews", "positive_pct", "negative_pct",
    ]]
    result = result.sort_values("total_reviews", ascending=False).reset_index(drop=True)
    return result


def dataset_fingerprint() -> str:
    """
    A cheap fingerprint of the data on disk (size + mtime of both CSVs),
    used to decide whether a cached model is stale and needs retraining.
    """
    parts = []
    for path in (HISTORICAL_PATH, USER_REVIEWS_PATH):
        if os.path.exists(path):
            stat = os.stat(path)
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime}")
        else:
            parts.append(f"{path}:missing")
    return "|".join(parts)
