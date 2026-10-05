"""
pages/1_Movie_Analysis.py
---------------------------
The core prediction workflow:
User enters movie name + review -> preprocess -> TF-IDF -> Logistic
Regression -> positive/negative + confidence -> save to user_reviews.csv
"""

import streamlit as st

from utils.ui import load_css, page_header, result_card
from utils.data_handler import (
    load_historical_data,
    save_user_review,
    dataset_fingerprint,
)
from utils.preprocessing import is_valid_review
from utils.model import train_model, save_artifacts, load_artifacts, predict_sentiment

st.set_page_config(page_title="Movie Analysis", page_icon="🎬", layout="wide")
load_css()

page_header("🎬 Movie Analysis", "Enter a movie name and review to predict its sentiment.")


@st.cache_resource(show_spinner=False)
def get_trained_artifacts(fingerprint: str):
    """
    Cached for the lifetime of the fingerprint. If the on-disk data
    changes (new user reviews saved, or movie_reviews.csv edited), the
    fingerprint changes and this function reruns automatically.
    """
    cached = load_artifacts(fingerprint)
    if cached is not None:
        return cached

    df = load_historical_data()
    result = train_model(df)
    save_artifacts(result, fingerprint)
    return result


# ---- Load / train model ----
try:
    with st.spinner("Preparing the sentiment model..."):
        fingerprint = dataset_fingerprint()
        artifacts = get_trained_artifacts(fingerprint)
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Something went wrong while preparing the model: {e}")
    st.stop()

vectorizer = artifacts["vectorizer"]
model = artifacts["model"]

st.markdown(
    f"""
    <div class="app-card fade-in">
        <p style="margin:0;color:var(--text-secondary);">
        Model ready — trained on <b style="color:var(--text-primary);">{artifacts['n_train']}</b> reviews,
        validated on <b style="color:var(--text-primary);">{artifacts['n_test']}</b> reviews.
        Current test accuracy: <b style="color:var(--accent);">{artifacts['accuracy']*100:.2f}%</b>
        (see 🤖 Model Performance for full details).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Input form ----
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

with st.form("review_form", clear_on_submit=False):
    movie_name = st.text_input(
        "🎬 Enter Movie Name",
        placeholder="e.g. Dangal",
    )
    review_text = st.text_area(
        "📝 Enter your movie review",
        placeholder=(
            "The movie was absolutely amazing. The acting was excellent "
            "and the story was very emotional."
        ),
        height=160,
    )
    submitted = st.form_submit_button("🔍 Analyze Review")

if submitted:
    if not movie_name or not movie_name.strip():
        st.warning("⚠️ Please enter a movie name before analyzing.")
    elif not is_valid_review(review_text):
        st.warning("⚠️ Please enter a non-empty review before analyzing.")
    else:
        try:
            with st.spinner("Analyzing your review..."):
                prediction = predict_sentiment(vectorizer, model, review_text)
                save_user_review(movie_name, review_text, prediction["label"])
            st.session_state.last_prediction = {
                "movie_name": movie_name.strip(),
                "prediction": prediction,
            }
        except Exception as e:
            st.error(f"❌ Could not analyze this review: {e}")

# ---- Result display ----
if st.session_state.last_prediction:
    pred = st.session_state.last_prediction["prediction"]
    movie = st.session_state.last_prediction["movie_name"]

    result_card(pred["label"], pred["confidence"], movie_name=movie)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">🟢 Positive</div>
            <div class="metric-value" style="color:var(--positive);">{pred['positive_pct']:.2f}%</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">🔴 Negative</div>
            <div class="metric-value" style="color:var(--negative);">{pred['negative_pct']:.2f}%</div></div>""",
            unsafe_allow_html=True,
        )

    