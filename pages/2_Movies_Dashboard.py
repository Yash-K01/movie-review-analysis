"""
pages/2_Movies_Dashboard.py
------------------------------
Per-movie sentiment statistics, cards, search, and Plotly charts.
Combines historical + user-submitted reviews so charts update
automatically as new reviews are analyzed.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.ui import load_css, page_header, movie_card_html
from utils.data_handler import get_combined_dataset, get_movie_stats

st.set_page_config(page_title="Movies Dashboard", page_icon="📊", layout="wide")
load_css()

page_header("📊 Movies Dashboard", "Sentiment statistics across every movie in the dataset.")

try:
    df = get_combined_dataset()
    stats = get_movie_stats(df)
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Could not load dashboard data: {e}")
    st.stop()

if stats.empty:
    st.info("No review data available yet.")
    st.stop()

# ---- Search ----
search = st.text_input("🔍 Search Movie", placeholder="Type a movie name...")
filtered = stats
if search and search.strip():
    filtered = stats[stats["movie_name"].str.contains(search.strip(), case=False, na=False)]
    if filtered.empty:
        st.warning(f"No movies found matching '{search}'.")

# ---- Top-level metrics ----
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Total Movies</div>
        <div class="metric-value">{stats.shape[0]}</div></div>""",
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Total Reviews</div>
        <div class="metric-value">{int(stats['total_reviews'].sum())}</div></div>""",
        unsafe_allow_html=True,
    )
with m3:
    overall_pos = (stats["positive_reviews"].sum() / stats["total_reviews"].sum() * 100) if stats["total_reviews"].sum() else 0
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Overall Positive Rate</div>
        <div class="metric-value" style="color:var(--positive);">{overall_pos:.2f}%</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ---- Movie cards ----
st.markdown('<div class="section-title fade-in">🎬 Movie Cards</div>', unsafe_allow_html=True)
if filtered.empty:
    st.stop()

cards_to_show = filtered.head(24)
cols = st.columns(3)
for idx, (_, row) in enumerate(cards_to_show.iterrows()):
    with cols[idx % 3]:
        st.markdown(movie_card_html(row), unsafe_allow_html=True)

if filtered.shape[0] > 24:
    st.caption(f"Showing 24 of {filtered.shape[0]} matching movies. Narrow your search to see others.")

st.markdown("---")

# ---- Charts ----
chart_data = filtered.sort_values("total_reviews", ascending=False).head(20)

st.markdown('<div class="section-title fade-in">📈 Positive vs Negative Reviews</div>', unsafe_allow_html=True)
fig1 = go.Figure()
fig1.add_trace(go.Bar(x=chart_data["movie_name"], y=chart_data["positive_pct"], name="Positive %", marker_color="#2ecc71"))
fig1.add_trace(go.Bar(x=chart_data["movie_name"], y=chart_data["negative_pct"], name="Negative %", marker_color="#ff5c6c"))
fig1.update_layout(barmode="group", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=420, legend_title_text="")
st.plotly_chart(fig1, use_container_width=True)

st.markdown('<div class="section-title fade-in">📊 Review Distribution (Stacked)</div>', unsafe_allow_html=True)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=chart_data["movie_name"], y=chart_data["positive_reviews"], name="Positive", marker_color="#2ecc71"))
fig2.add_trace(go.Bar(x=chart_data["movie_name"], y=chart_data["negative_reviews"], name="Negative", marker_color="#ff5c6c"))
fig2.update_layout(barmode="stack", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=420, legend_title_text="")
st.plotly_chart(fig2, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="section-title fade-in">🏆 Most Positively Reviewed</div>', unsafe_allow_html=True)
    top_pos = filtered.sort_values("positive_pct", ascending=False).head(10)
    fig3 = px.bar(top_pos, x="positive_pct", y="movie_name", orientation="h",
                  color_discrete_sequence=["#2ecc71"])
    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=400, yaxis={"categoryorder": "total ascending"}, xaxis_title="Positive %", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title fade-in">⚠️ Most Negatively Reviewed</div>', unsafe_allow_html=True)
    top_neg = filtered.sort_values("negative_pct", ascending=False).head(10)
    fig4 = px.bar(top_neg, x="negative_pct", y="movie_name", orientation="h",
                  color_discrete_sequence=["#ff5c6c"])
    fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=400, yaxis={"categoryorder": "total ascending"}, xaxis_title="Negative %", yaxis_title="")
    st.plotly_chart(fig4, use_container_width=True)
