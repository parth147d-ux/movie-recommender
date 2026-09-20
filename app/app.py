"""
Streamlit UI for the movie recommender.

Run from the project root:
    python -m streamlit run app/app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model, weighted_rating
from src.preprocess import preprocess
from src.recommend import recommend, suggest_titles

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


# Cached so the dataset isn't re-parsed and the similarity matrix isn't
# rebuilt on every interaction. Without these, each click re-ran the whole
# pipeline (several seconds and ~5000x5000 of wasted work per press).
@st.cache_data(show_spinner="Loading and preparing the dataset...")
def load_df():
    return preprocess()


@st.cache_resource(show_spinner="Building the similarity model...")
def build_model(method):
    df = load_df()
    return create_model(df, method=method)


@st.cache_data(show_spinner=False)
def build_quality():
    return weighted_rating(load_df())


st.title("🎬 Movie Recommender System")
st.caption(
    "Content-based recommendations using TF-IDF over movie tags "
    "(plot, genres, keywords, top cast, director) and cosine similarity."
)

with st.sidebar:
    st.header("Settings")
    method = st.radio(
        "Vectorizer",
        ["tfidf", "count"],
        format_func=lambda m: {"tfidf": "TF-IDF", "count": "Bag of words"}[m],
        help="TF-IDF down-weights words that appear across many movies.",
    )
    top_n = st.slider("Number of recommendations", 3, 15, 5)
    quality_weight = st.slider(
        "Quality weight", 0.0, 0.6, 0.25, 0.05,
        help="0 = pure content similarity. Higher values favour "
             "better-rated, well-voted films among similar candidates.",
    )

try:
    df = load_df()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

similarity = build_model(method)
quality = build_quality()

st.write(f"**{len(df):,}** movies loaded.")

tab_search, tab_pick = st.tabs(["Search by name", "Pick from list"])

with tab_search:
    typed = st.text_input(
        "Enter a movie name",
        placeholder="e.g. dark knigth — typos and lowercase are fine",
    )
    go_typed = st.button("Recommend", key="btn_typed", type="primary")

with tab_pick:
    picked = st.selectbox(
        "Choose a movie",
        options=sorted(df["title"].tolist()),
        index=None,
        placeholder="Start typing to filter...",
    )
    go_picked = st.button("Recommend", key="btn_picked", type="primary")

query = None
if go_typed and typed.strip():
    query = typed
elif go_picked and picked:
    query = picked

if query:
    results, matched = recommend(
        query, df, similarity,
        top_n=top_n, quality_weight=quality_weight, quality=quality,
    )

    if matched is None:
        st.warning(f"Couldn't find a movie matching “{query}”.")
        tips = suggest_titles(query, df)
        if tips:
            st.write("Did you mean:")
            for t in tips:
                st.write(f"- {t}")
    else:
        if matched.lower() != query.strip().lower():
            st.info(f"Showing results for **{matched}**")
        st.subheader("Recommended movies")
        for rank, r in enumerate(results, start=1):
            year = f" ({r['year']})" if r["year"] else ""
            with st.container(border=True):
                st.markdown(f"**{rank}. {r['title']}**{year}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rating", f"{r['rating']}/10")
                c2.metric("Votes", f"{r['votes']:,}")
                c3.metric("Similarity", f"{r['similarity']:.3f}")
elif go_typed or go_picked:
    st.warning("Please enter or select a movie first.")
