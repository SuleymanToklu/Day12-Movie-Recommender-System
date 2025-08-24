import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

LANGUAGES = {
    "tr": {
        "app_title": "🎬 Hibrit Film Tavsiye Sistemi",
        "app_subtitle": "Bir film seçin, size hibrit modelimizle en iyi önerileri sunalım!",
        "settings_header": "Ayarlar",
        "select_movie_label": "Başlamak için bir film seçin:",
        "num_recs_label": "Tavsiye Sayısı",
        "rec_type_label": "Tavsiye Türü",
        "rec_type_hybrid": "✨ Hibrit",
        "rec_type_content": "🧠 İçerik Tabanlı",
        "rec_type_collab": "👥 Benzer Zevkler (SVD)",
        "get_recs_button": "Tavsiyeleri Getir",
        "spinner_text": "🎬 Filmler sizin için hazırlanıyor...",
        "recommendations_for": "için öneriler:",
        "no_recs_found": "Bu film için tavsiye bulunamadı. Lütfen başka bir film deneyin.",
        "welcome_message": "Lütfen bir film seçip 'Tavsiyeleri Getir' butonuna tıklayın.",
        "tab_recommender": "🤖 Tavsiye Aracı",
        "tab_about": "📖 Proje Hakkında",
        "about_header": "Projenin Amacı ve Teknik Detaylar",
        "about_intro": "Bu proje, #30DaysOfAI challenge'ı kapsamında geliştirilmiş, iki farklı yapay zeka tekniğini birleştiren bir hibrit film tavsiye sistemidir.",
        "about_tech_stack_header": "Kullanılan Teknolojiler",
        "about_models_header": "Tavsiye Modelleri",
        "about_content_model_title": "İçerik Tabanlı Filtreleme",
        "about_content_model_desc": "Bu model, filmlerin metinsel özelliklerine (özet, tür, anahtar kelimeler) dayanır. Birbirine benzeyen filmleri bulmak için **TF-IDF** vektörleştirmesi ve anlık **Kosinüs Benzerliği** hesaplaması kullanır.",
        "about_collab_model_title": "İşbirlikçi Filtreleme (Scikit-learn ile)",
        "about_collab_model_desc": "Bu model, kullanıcıların geçmiş puanlarını analiz eder. **TruncatedSVD** algoritması kullanarak filmleri gizli faktör uzayında temsil eder ve bu uzaydaki yakınlıklarına göre benzer filmleri bulur.",
        "about_hybrid_model_title": "Hibrit Model",
        "about_hybrid_model_desc": "Varsayılan 'Hibrit' seçeneği, yukarıdaki iki modelden gelen sonuçları birleştirerek daha zengin ve isabetli bir tavsiye listesi sunar.",
        "language_select": "Dil / Language"
    },
    "en": {
        "app_title": "🎬 Hybrid Movie Recommender System",
        "app_subtitle": "Select a movie and let our hybrid model recommend the best matches for you!",
        "settings_header": "Settings",
        "select_movie_label": "Select a movie to start:",
        "num_recs_label": "Number of Recommendations",
        "rec_type_label": "Recommendation Type",
        "rec_type_hybrid": "✨ Hybrid",
        "rec_type_content": "🧠 Content-Based",
        "rec_type_collab": "👥 Similar Tastes (SVD)",
        "get_recs_button": "Get Recommendations",
        "spinner_text": "🎬 Preparing your movies...",
        "recommendations_for": "Recommendations for:",
        "no_recs_found": "No recommendations found for this movie. Please try another one.",
        "welcome_message": "Please select a movie and click the 'Get Recommendations' button.",
        "tab_recommender": "🤖 Recommender Tool",
        "tab_about": "📖 About the Project",
        "about_header": "Project Purpose and Technical Details",
        "about_intro": "This project is a hybrid movie recommender system developed for the #30DaysOfAI challenge, combining two different AI techniques.",
        "about_tech_stack_header": "Technology Stack",
        "about_models_header": "Recommendation Models",
        "about_content_model_title": "Content-Based Filtering",
        "about_content_model_desc": "This model relies on the textual attributes of movies (overview, genre, keywords). It uses **TF-IDF** vectorization and on-the-fly **Cosine Similarity** calculation to find similar movies.",
        "about_collab_model_title": "Collaborative Filtering (with Scikit-learn)",
        "about_collab_model_desc": "This model analyzes past user ratings. It uses the **TruncatedSVD** algorithm to represent movies in a latent factor space and finds similar movies based on their proximity in this space.",
        "about_hybrid_model_title": "Hybrid Model",
        "about_hybrid_model_desc": "The default 'Hybrid' option combines the results from the two models above to provide a richer and more accurate list of recommendations.",
        "language_select": "Dil / Language"
    }
}

st.set_page_config(page_title="Film Tavsiye Sistemi", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

if 'lang' not in st.session_state:
    st.session_state.lang = 'tr'

def T(key):
    return LANGUAGES[st.session_state.lang].get(key, key)

@st.cache_resource
def load_resources():
    try:
        svd_model = joblib.load('svd_model.pkl')
        tfidf_matrix = joblib.load('tfidf_matrix.pkl')
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
    sim_scores = sorted(list(enumerate(cosine_sims)), key=lambda x: x[1], reverse=True)[1:num_recs+1]
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

st.markdown("""
<style>
    .movie-card {
        background-color: #2c2c2c; border: 1px solid #444; border-radius: 10px;
        padding: 1rem; color: white; margin-bottom: 20px; height: 100%;
        display: flex; flex-direction: column; justify-content: space-between;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover { transform: scale(1.05); box-shadow: 0 8px 30px rgba(255, 255, 255, 0.1); }
    .movie-title { font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem; color: #1DB954; }
</style>
""", unsafe_allow_html=True)

st.title(T("app_title"))
st.write(T("app_subtitle"))

if any(res is None for res in [svd_model, tfidf_matrix, movies_df, movie_ids_map]):
    st.error("Model dosyaları bulunamadı! Lütfen önce `train_model.py` script'ini çalıştırın.")
    st.stop()

with st.sidebar:
    st.header(T("settings_header"))
    selected_lang_display = st.radio(T("language_select"), options=['Türkçe', 'English'], horizontal=True, index=0 if st.session_state.lang == 'tr' else 1)
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
                if recommendation_type == T("rec_type_hybrid"):
                    content_recs = get_content_recommendations(selected_movie_title, tfidf_matrix, movies_df, num_recommendations)
                    collab_recs = get_collaborative_recommendations(selected_movie_id, svd_model, movie_ids_map, num_recommendations)
                    recs_set = set()
                    recommendations = [rec for rec in content_recs + collab_recs if rec not in recs_set and not recs_set.add(rec)]

                elif recommendation_type == T("rec_type_content"):
                    recommendations = get_content_recommendations(selected_movie_title, tfidf_matrix, movies_df, num_recommendations)

                elif recommendation_type == T("rec_type_collab"):
                     recommendations = get_collaborative_recommendations(selected_movie_id, svd_model, movie_ids_map, num_recommendations)
            
            if not recommendations:
                st.warning(T("no_recs_found"))
            else:
                final_recs = [rec for rec in recommendations if rec != selected_movie_id][:num_recommendations]
                cols = st.columns(5)
                for i, movie_id in enumerate(final_recs):
                    movie_details = movies_df[movies_df['movieId'] == movie_id]
                    if not movie_details.empty:
                        with cols[i % 5]:
                            st.markdown(f"""
                            <div class="movie-card">
                                <p class="movie-title">{movie_details.iloc[0]['title']}</p>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.info(T("welcome_message"))

with tab2:
    st.header(T("about_header"))
    st.markdown(T("about_intro"))

    st.subheader(T("about_tech_stack_header"))
    st.markdown("""
    - **Python** & **Streamlit**
    - **Pandas** & **NumPy**
    - **Scikit-learn** (TF-IDF & TruncatedSVD)
    - **Scipy**
    """)

    st.subheader(T("about_models_header"))
    st.info(f"**{T('about_content_model_title')}**\n\n{T('about_content_model_desc')}")
    st.info(f"**{T('about_collab_model_title')}**\n\n{T('about_collab_model_desc')}")
    st.success(f"**{T('about_hybrid_model_title')}**\n\n{T('about_hybrid_model_desc')}")