# 🎬 Movie Review Analysis System

A supervised-machine-learning web app that predicts whether a movie
review is **positive** or **negative**, built with **TF-IDF +
Logistic Regression** and a **Streamlit** UI.

---

## 1. Description

Enter a movie name and a review, and the app predicts the sentiment
(positive/negative) with a confidence score, using a Logistic
Regression model trained on TF-IDF features from real IMDB movie
reviews. Every analyzed review is saved and a **Movies Dashboard**
shows sentiment statistics per movie with interactive Plotly charts.

## 2. Objective

1. Load historical review data from CSV.
2. Preprocess review text.
3. Convert text to numeric features with TF-IDF.
4. Train a Logistic Regression classifier.
5. Calculate & display real accuracy/precision/recall/F1.
6. Predict sentiment for user-submitted reviews with confidence.
7. Save every analyzed review to `user_reviews.csv`.
8. Combine historical + user data for future retraining.
9. Show per-movie sentiment stats on a searchable dashboard.

## 3. Features

- 🎬 **Movie Analysis** — enter a movie + review, get an instant
  Positive/Negative prediction with confidence %.
- 📊 **Movies Dashboard** — searchable movie cards + 4 Plotly charts
  (grouped bar, stacked bar, top-positive, top-negative).
- 🤖 **Model Performance** — live accuracy, precision, recall, F1,
  confusion matrix, and full classification report — all computed
  from the actual train/test split, never hardcoded.
- ℹ️ **About** — explains TF-IDF, Logistic Regression, supervised
  learning, and two important data-honesty notes (see below).
- 💾 Every analyzed review is appended to `user_reviews.csv` for
  future retraining experiments.
- 🎨 Modern, responsive, dark-themed UI with hover effects and
  fade-in animations; neutral palette with green/red reserved for
  sentiment only.

## 4. Technologies Used

| Purpose | Library |
|---|---|
| Language | Python |
| ML | scikit-learn (TF-IDF Vectorizer, Logistic Regression) |
| Data | pandas, NumPy |
| Web app | Streamlit |
| Charts | Plotly |
| Storage | CSV files |
| Model persistence | joblib |

## 5. Machine Learning Workflow

```
Historical CSV Dataset
        ↓
Data Cleaning (utils/data_handler.py)
        ↓
Text Preprocessing (utils/preprocessing.py)
        ↓
Train/Test Split (80/20, stratified, random_state=42)
        ↓
TF-IDF Vectorization (fit on TRAIN only — no leakage)
        ↓
Logistic Regression (max_iter=1000)
        ↓
Evaluation: accuracy, precision, recall, F1, confusion matrix
        ↓
User Review Prediction (predict_proba → confidence %)
        ↓
Save to user_reviews.csv → available for future retraining
```

The model and vectorizer are cached with `joblib` under `models/`
and automatically retrained whenever the underlying CSV data changes
(tracked via a size+mtime fingerprint), so the app doesn't retrain
on every rerun but never serves a stale model either.

## 6. Dataset Format

`data/movie_reviews.csv` and `data/user_reviews.csv` both use:

```csv
movie_name,review,sentiment
Dangal,Excellent acting and emotional story,positive
Dangal,I did not enjoy the movie at all,negative
```

### ⚠️ Data honesty notes — please read

1. **Movie names are synthetic.** This project ships with the real
   **IMDB 50K review dataset** you uploaded (50,000 reviews, real
   text, real positive/negative labels). That dataset has **no movie
   titles** in it — it's anonymized. To power the per-movie
   dashboard the spec asked for, each review was **randomly (but
   reproducibly, `numpy` seed 42) assigned** one of 40 well-known
   movie titles. The review text and sentiment labels driving the
   model are 100% real; only `movie_name` is a demo stand-in. If you
   have real per-movie IMDB data, just swap `data/movie_reviews.csv`
   for a file with genuine `movie_name` values — no code changes
   needed.
2. **User-submitted labels are predictions, not verified truth.**
   `user_reviews.csv` stores the **model's predicted** sentiment for
   each review a user analyzes — not a human-confirmed label. Using
   that file to retrain the model is fine for experimentation, but
   it can reinforce the model's existing mistakes/bias. A production
   system should let a human confirm or correct each label (e.g. a
   👍/👎 or an editable dropdown) before it's used for retraining.

## 7. Project Structure

```
movie-review-analysis/
│
├── app.py                        # Home page
│
├── pages/
│   ├── 1_Movie_Analysis.py       # Prediction workflow
│   ├── 2_Movies_Dashboard.py     # Per-movie stats + charts + search
│   ├── 3_Model_Performance.py    # Accuracy, confusion matrix, report
│   └── 4_About.py                # Explanations + data honesty notes
│
├── data/
│   ├── movie_reviews.csv         # Historical data (IMDB + synthetic movie names)
│   └── user_reviews.csv          # Auto-created; appended to at runtime
│
├── models/                       # Auto-created; cached model + vectorizer
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── training_meta.pkl
│
├── utils/
│   ├── preprocessing.py          # preprocess_text(), is_valid_review()
│   ├── model.py                  # train_model(), predict_sentiment(), persistence
│   ├── data_handler.py           # CSV I/O, combining datasets, movie stats
│   └── ui.py                     # Shared CSS/HTML helpers
│
├── assets/
│   └── style.css                 # Shared stylesheet (cards, animations, theme)
│
├── requirements.txt
└── README.md
```

## 8. Installation

```bash
cd movie-review-analysis
pip install -r requirements.txt
```

## 9. Running the App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).
Use the sidebar to navigate between **Home**, **Movie Analysis**,
**Movies Dashboard**, **Model Performance**, and **About**.

## 10. Model Evaluation

On the bundled IMDB dataset (50,000 reviews, 80/20 stratified split),
the Logistic Regression + TF-IDF pipeline achieves **~89% accuracy**
on the held-out test set (exact figure shown live on the Model
Performance page — it is computed at run time, never hardcoded).

## 11. Screenshots

_Add screenshots here after running the app locally:_

- `screenshots/home.png`
- `screenshots/movie-analysis.png`
- `screenshots/dashboard.png`
- `screenshots/model-performance.png`

## 12. Future Enhancements

- Let users confirm/correct the predicted label before it's used for
  retraining (turns `user_reviews.csv` into true ground truth).
- Replace synthetic movie-name assignment with a real per-movie IMDB
  dataset if one becomes available.
- Add a "Retrain now" button on the Model Performance page.
- Support multi-class ratings (1–5 stars) instead of binary sentiment.
- Add model comparison (Logistic Regression vs. Naive Bayes vs. SVM).
- Deploy to Streamlit Community Cloud / Docker.

---

Built as a supervised-learning student/portfolio project: clean
separation of ML logic (`utils/model.py`), data logic
(`utils/data_handler.py`), and UI logic (`pages/*.py`, `utils/ui.py`)
so it's easy to explain end-to-end in a viva.
