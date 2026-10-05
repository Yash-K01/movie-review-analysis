"""
app.py
-------
Home / landing page for the Movie Review Analysis System.
Run with:  streamlit run app.py
"""

import streamlit as st
from utils.ui import load_css, page_header

st.set_page_config(
    page_title="Movie Review Analysis System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

page_header(
    "🎬 Movie Review Analysis System",
    "Analyze movie reviews using Supervised Machine Learning",
)

st.markdown(
    """
    <div class="app-card fade-in">
        <p style="margin:0; color:var(--text-primary); font-size:1rem;">
        This project classifies a movie review as <b style="color:var(--positive);">Positive</b>
        or <b style="color:var(--negative);">Negative</b> using a
        <b>TF-IDF + Logistic Regression</b> pipeline trained on real IMDB movie reviews,
        then lets you explore per-movie sentiment statistics on the Dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title fade-in">What you can do here</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
features = [
    ("🤖", "AI Sentiment Analysis", "Predict positive or negative movie reviews from raw text."),
    ("📈", "Model Accuracy", "Evaluate real Logistic Regression performance on held-out test data."),
    ("🎬", "Movie Insights", "Explore sentiment statistics for every movie in the dataset."),
    ("💾", "Continuous Data Collection", "Every analyzed review is stored for future retraining."),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
    with col:
        st.markdown(
            f"""
            <div class="app-card fade-in">
                <h3>{icon} {title}</h3>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown('<div class="section-title fade-in">Get started</div>', unsafe_allow_html=True)
st.page_link("pages/1_Movie_Analysis.py", label="🎬 Analyze a Review  →", icon=None)
st.page_link("pages/2_Movies_Dashboard.py", label="📊 Explore the Movies Dashboard  →", icon=None)

st.write("")
st.info(
    "ℹ️ **About this dataset:** the reviews and sentiment labels come from the real "
    "IMDB 50K review dataset. That dataset does not include movie titles, so a movie "
    "name has been assigned to each review from a fixed list for demonstration "
    "purposes on the Movies Dashboard — see the **About** page for details.",
)

st.markdown("---")
st.caption("Navigate using the sidebar: 🎬 Movie Analysis · 📊 Movies Dashboard ·")
