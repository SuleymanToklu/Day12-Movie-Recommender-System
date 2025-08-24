import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

LANGUAGES = { "tr": { "app_title": "🎬 Film Tavsiye Sistemi (Optimize Edilmiş)" }, "en": { "app_title": "🎬 Movie Recommender (Optimized)" } } 

st.set_page_config(page_title="Film Tavsiye Sistemi", page_icon="🎬", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'tr'

def T(key):
    return LANGUAGES[st.session_state.lang].get(key, key)

@st.cache_resource
def load_resources():
    try:
        svd_model = joblib.load('svd_model.pkl')
        tfidf_matrix = joblib.load('tfidf_matrix.pkl') # Değişti
        movies_df = joblib.load('movies_df.pkl')
        movie_ids_map = joblib.load('movie_ids_map.pkl')
        return svd_model, tfidf_matrix, movies_df, movie_ids_map
    except FileNotFoundError:
        return None, None, None, None

svd_model, tfidf_matrix, movies_df, movie_ids_map = load_resources()

def get_content_recommendations(movie_title, tfidf_matrix, movies_df, num_recs=10):
    filtered_movies = movies_df[movies_df['title'] == movie_title]
    if filtered_movies.empty: return []
    
    idx = filtered_movies.index[0]
    
    cosine_sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sims))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recs+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['movieId'].iloc[movie_indices].tolist()

def get_collaborative_recommendations(movie_id, svd_model, movie_ids_map, num_recs=10):
    try:
        movie_idx = movie_ids_map.get_loc(movie_id)
        movie_vector = svd_model.components_[:, movie_idx].reshape(1, -1)
        all_movie_vectors = svd_model.components_.T
        sims = cosine_similarity(movie_vector, all_movie_vectors)
        similar_indices = np.argsort(sims[0])[::-1][1:num_recs+1]
        similar_movie_ids = movie_ids_map[similar_indices]
        return similar_movie_ids.tolist()
    except (KeyError, IndexError):
        return []

st.title(T("app_title"))

if svd_model is None or tfidf_matrix is None or movies_df is None:
    st.error("Model dosyaları bulunamadı! Lütfen önce `train_model.py` script'ini çalıştırın.")
    st.stop()

with st.sidebar:
    selected_movie_title = st.selectbox("Bir film seçin:", options=sorted(movies_df['title'].dropna().unique()))
    num_recommendations = st.slider('Tavsiye Sayısı', 5, 20, 10)

if st.sidebar.button('Tavsiyeleri Getir'):
    if selected_movie_title:
        selected_movie_id = movies_df[movies_df['title'] == selected_movie_title].iloc[0]['movieId']
        
        st.subheader(f"'{selected_movie_title}' için öneriler:")
        
        content_recs = get_content_recommendations(selected_movie_title, tfidf_matrix, movies_df, num_recommendations)
        
        if not content_recs:
            st.warning("Bu film için tavsiye bulunamadı.")
        else:
            st.write([movies_df[movies_df['movieId'] == mid]['title'].iloc[0] for mid in content_recs if not movies_df[movies_df['movieId'] == mid].empty])