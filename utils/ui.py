"""
ui.py
------
Small shared UI helpers (CSS loading, reusable card renderers) so the
individual Streamlit pages stay short and focused on layout/flow
rather than raw HTML.
"""

import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")


def load_css():
    """Inject the shared stylesheet once per page render."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="hero-title fade-in">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="hero-subtitle fade-in">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card fade-in">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_card(label: str, confidence: float, movie_name: str = ""):
    is_positive = label == "positive"
    css_class = "positive" if is_positive else "negative"
    emoji = "🟢" if is_positive else "🔴"
    headline = "POSITIVE REVIEW" if is_positive else "NEGATIVE REVIEW"
    movie_bit = f" for <b>{movie_name}</b>" if movie_name else ""

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-headline">{emoji} {headline}</div>
            <div class="result-sub">Your review{movie_bit} is predicted as {label.capitalize()}.</div>
            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def movie_card_html(row) -> str:
    return f"""
    <div class="movie-card fade-in">
        <h4>🎬 {row['movie_name']}</h4>
        <div class="movie-stat-row"><span class="label">Total Reviews</span><span>{int(row['total_reviews'])}</span></div>
        <div class="movie-stat-row"><span class="pct-positive">🟢 Positive</span><span class="pct-positive">{row['positive_pct']:.2f}%</span></div>
        <div class="movie-stat-row"><span class="pct-negative">🔴 Negative</span><span class="pct-negative">{row['negative_pct']:.2f}%</span></div>
    </div>
    """
