import streamlit as st
import pandas as pd
from src.serving.inference import recommend_for_user, similar_movies_for_id, get_model_metrics, get_genre_popularity, get_sentiment_trends

st.title("🎬 FilmFusion Interactive Recommender Dashboard")

tab1, tab2, tab3 = st.tabs(["User Recommendations", "Model Insights", "Genre & Sentiment"])

with tab1:
    user_id = st.number_input("Enter a User ID:", min_value=1, step=1)
    n_recs = st.slider("Number of Recommendations", 1, 20, 10)
    if st.button("Get Recommendations") and user_id:
        recs = recommend_for_user(user_id, n_recs)
        if recs:
            st.write(pd.DataFrame(recs))
        else:
            st.warning("No recommendations found for this user.")

    movie_id = st.number_input("Enter a Movie ID to find similar:", min_value=1, step=1)
    n_sims = st.slider("Number of Similar Movies", 1, 20, 10)
    if st.button("Find Similar Movies") and movie_id:
        sims = similar_movies_for_id(movie_id, n_sims)
        if sims:
            st.write(pd.DataFrame(sims))
        else:
            st.warning("No similar movies found.")

with tab2:
    st.header("Key Model Performance Metrics")
    metrics = get_model_metrics()  # e.g., loads metrics from last eval
    st.json(metrics)

with tab3:
    st.header("Genre Popularity")
    genres = get_genre_popularity()
    st.bar_chart(genres)

    st.header("Sentiment Trends")
    sentiments = get_sentiment_trends()
    st.line_chart(sentiments)
