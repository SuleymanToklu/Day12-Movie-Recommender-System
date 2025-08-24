import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

LANGUAGES = {
    "tr": { "app_title": "🎬 Film Tavsiye Sistemi (SKLearn Versiyonu)", "app_subtitle": "Bir film seçin, size hibrit modelimizle en iyi önerileri sunalım!", "settings_header": "Ayarlar", "select_movie_label": "Başlamak için bir film seçin:", "num_recs_label": "Tavsiye Sayısı", "rec_type_label": "Tavsiye Türü", "rec_type_hybrid": "✨ Hibrit", "rec_type_content": "🧠 İçerik Tabanlı", "rec_type_collab": "👥 Benzer Zevkler (SVD)", "get_recs_button": "Tavsiyeleri Getir", "spinner_text": "🎬 Filmler sizin için hazırlanıyor...", "recommendations_for": "için öneriler:", "no_recs_found": "Bu film için tavsiye bulunamadı. Lütfen başka bir film deneyin.", "welcome_message": "Lütfen bir film seçip 'Tavsiyeleri Getir' butonuna tıklayın.", "tab_recommender": "🤖 Tavsiye Aracı", "tab_about": "📖 Proje Hakkında", "about_hybrid_model_desc": "... Bu versiyon, SVD modeli için endüstri standardı olan scikit-learn kütüphanesini kullanmaktadır." },
    "en": { "app_title": "🎬 Movie Recommender (SKLearn Version)", "app_subtitle": "Select a movie and let our hybrid model recommend the best matches!", "settings_header": "Settings", "select_movie_label": "Select a movie to start:", "num_recs_label": "Number of Recommendations", "rec_type_label": "Recommendation Type", "rec_type_hybrid": "✨ Hybrid", "rec_type_content": "🧠 Content-Based", "rec_type_collab": "👥 Similar Tastes (SVD)", "get_recs_button": "Get Recommendations", "spinner_text": "🎬 Preparing your movies...", "recommendations_for": "Recommendations for:", "no_recs_found": "No recommendations found for this movie. Please try another one.", "welcome_message": "Please select a movie and click the 'Get Recommendations' button.", "tab_recommender": "🤖 Recommender Tool", "tab_about": "📖 About the Project", "about_hybrid_model_desc": "... This version uses the industry-standard scikit-learn library for the SVD model." }
}

st.set_page_config(page_title="Film Tavsiye Sistemi", page_icon="🎬", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = 'tr'

def T(key):
    return LANGUAGES[st.session_state.lang].get(key, key)

@st.cache_resource
def load_resources():
    try:
        svd_model = joblib.load('svd_model.pkl')
        cosine_sim = joblib.load('cosine_sim.pkl')
        movies_df = joblib.load('movies_df.pkl')
        movie_ids_map = joblib.load('movie_ids_map.pkl')
        return svd_model, cosine_sim, movies_df, movie_ids_map
    except FileNotFoundError:
        return None, None, None, None

svd_model, cosine_sim, movies_df, movie_ids_map = load_resources()

def get_content_recommendations(movie_title, cosine_sim, movies_df, num_recs=10):
    filtered_movies = movies_df[movies_df['title'] == movie_title]
    if filtered_movies.empty: return []
    idx = filtered_movies.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
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

with st.sidebar:
    st.header(T("settings_header"))
    selected_lang_display = st.radio(T("language_select"), options=['Türkçe', 'English'], horizontal=True)
    st.session_state.lang = 'tr' if selected_lang_display == 'Türkçe' else 'en'
    
    movie_titles = sorted(movies_df['title'].dropna().unique())
    selected_movie_title = st.selectbox(T("select_movie_label"), options=movie_titles)
    num_recommendations = st.slider(T("num_recs_label"), 5, 20, 10)
    recommendation_type = st.radio(T("rec_type_label"), (T("rec_type_hybrid"), T("rec_type_content"), T("rec_type_collab")))

tab1, tab2 = st.tabs([T("tab_recommender"), T("tab_about")])

with tab1:
    if st.sidebar.button(T("get_recs_button"), use_container_width=True):
        if selected_movie_title:
            selected_movie = movies_df[movies_df['title'] == selected_movie_title].iloc[0]
            selected_movie_id = selected_movie['movieId']
            st.subheader(f"'{selected_movie_title}' {T('recommendations_for')}")
            
            recommendations = []
            with st.spinner(T("spinner_text")):
                if 'Hibrit' in recommendation_type or 'Hybrid' in recommendation_type:
                    content_recs = get_content_recommendations(selected_movie_title, cosine_sim, movies_df, num_recommendations)
                    collab_recs = get_collaborative_recommendations(selected_movie_id, svd_model, movie_ids_map, num_recommendations)
                    # ... birleştirme mantığı ...
                    recs_set = set()
                    recommendations = [rec for rec in content_recs + collab_recs if rec not in recs_set and not recs_set.add(rec)]

                elif 'İçerik' in recommendation_type or 'Content' in recommendation_type:
                    recommendations = get_content_recommendations(selected_movie_title, cosine_sim, movies_df, num_recommendations)

                elif 'Benzer Zevkler' in recommendation_type or 'Similar Tastes' in recommendation_type:
                     recommendations = get_collaborative_recommendations(selected_movie_id, svd_model, movie_ids_map, num_recommendations)
            
            if not recommendations:
                st.warning(T("no_recs_found"))
            else:
                final_recs = [rec for rec in recommendations if rec != selected_movie_id][:num_recommendations]
                st.write([movies_df[movies_df['movieId'] == mid]['title'].iloc[0] for mid in final_recs if not movies_df[movies_df['movieId'] == mid].empty])