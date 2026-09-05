import streamlit as st
from src.preprocess import preprocess
from src.model import create_model
from src.recommend import recommend

st.title("🎬 Movie Recommender System")

df = preprocess()
similarity = create_model(df)

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    results = recommend(movie_name, df, similarity)
    
    st.write("### Recommended Movies:")
    for movie in results:
        st.write(movie)
        #cd movie-recommender
        #python -m streamlit run app/app.py