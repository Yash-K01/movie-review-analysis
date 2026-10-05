"""
model.py
---------
The supervised-learning pipeline: TF-IDF + Logistic Regression.

Pipeline:
    Historical CSV -> preprocess -> train/test split -> TF-IDF ->
    Logistic Regression -> accuracy / precision / recall / F1 /
    confusion matrix -> persisted model+vectorizer for reuse.

Nothing in this file touches Streamlit -- it's pure ML/data code so it
can be explained and tested independently of the UI.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils.preprocessing import preprocess_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
META_PATH = os.path.join(MODELS_DIR, "training_meta.pkl")


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Train TF-IDF + Logistic Regression on the given dataframe
    (must have 'review' and 'sentiment' columns) and evaluate on a
    held-out test split.

    Returns a dict containing the fitted vectorizer, fitted model,
    and every metric the UI needs to display -- nothing here is
    hardcoded; all numbers come from the actual train/test evaluation.
    """
    working = df.copy()
    working["clean_review"] = working["review"].apply(preprocess_text)
    working = working[working["clean_review"].str.len() > 0]

    if working["sentiment"].nunique() < 2:
        raise ValueError(
            "Cannot train: dataset contains only one sentiment class "
            "after cleaning."
        )

    X_text = working["clean_review"]
    y = working["sentiment"].map({"negative": 0, "positive": 1})

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit the vectorizer ONLY on training data to avoid leakage; the
    # same fitted vectorizer is reused to transform test data & any
    # future user review.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["negative", "positive"],
        zero_division=0, output_dict=True,
    )

    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_total": working.shape[0],
    }


def save_artifacts(result: dict, fingerprint: str) -> None:
    ensure_models_dir()
    joblib.dump(result["model"], MODEL_PATH)
    joblib.dump(result["vectorizer"], VECTORIZER_PATH)
    meta = {k: v for k, v in result.items() if k not in ("model", "vectorizer")}
    meta["fingerprint"] = fingerprint
    joblib.dump(meta, META_PATH)


def load_artifacts(expected_fingerprint: str):
    """
    Load a previously trained model+vectorizer+metrics ONLY if they
    match the current data fingerprint. Returns None if unavailable or
    stale, so the caller knows to retrain.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
            and os.path.exists(META_PATH)):
        return None
    try:
        meta = joblib.load(META_PATH)
        if meta.get("fingerprint") != expected_fingerprint:
            return None
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception:
        return None

    result = dict(meta)
    result["model"] = model
    result["vectorizer"] = vectorizer
    return result


def predict_sentiment(vectorizer, model, raw_text: str) -> dict:
    """
    Predict sentiment for a single raw review string.

    Returns:
        {
          "label": "positive" | "negative",
          "confidence": float (0-100, confidence in the predicted label),
          "positive_pct": float (0-100),
          "negative_pct": float (0-100),
        }
    """
    clean = preprocess_text(raw_text)
    X = vectorizer.transform([clean])
    proba = model.predict_proba(X)[0]  # [P(negative), P(positive)]
    negative_pct = float(proba[0]) * 100
    positive_pct = float(proba[1]) * 100

    label = "positive" if positive_pct >= negative_pct else "negative"
    confidence = max(positive_pct, negative_pct)

    return {
        "label": label,
        "confidence": confidence,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
    }
