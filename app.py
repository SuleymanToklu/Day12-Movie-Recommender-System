import streamlit as st
import pandas as pd
import joblib
import ast
import numpy as np
from collections import defaultdict

LANGUAGES = {
    "tr": {
        "app_title": "🎬 Gelişmiş Film Tavsiye Sistemi",
        "app_subtitle": "Bir film seçin, size hibrit modelimizle en iyi önerileri sunalım!",
        "settings_header": "Ayarlar",
        "select_movie_label": "Başlamak için bir film seçin:",
        "num_recs_label": "Tavsiye Sayısı",
        "rec_type_label": "Tavsiye Türü",
        "rec_type_hybrid": "✨ Hibrit",
        "rec_type_content": "🧠 İçerik Tabanlı",
        "rec_type_collab": "👥 İşbirlikçi Filtreleme",
        "get_recs_button": "Tavsiyeleri Getir",
        "spinner_text": "🎬 Filmler sizin için hazırlanıyor...",
        "recommendations_for": "için öneriler:",
        "no_recs_found": "Bu film için tavsiye bulunamadı. Lütfen başka bir film deneyin.",
        "welcome_message": "Lütfen bir film seçip 'Tavsiyeleri Getir' butonuna tıklayın.",
        "tab_recommender": "🤖 Tavsiye Aracı",
        "tab_about": "📖 Proje Hakkında",
        "about_header": "Projenin Amacı ve Teknik Detaylar",
        "about_intro": "Bu proje, #30DaysOfAI challenge'ının 12. Günü için geliştirilmiş bir hibrit film tavsiye sistemidir. Amaç, kullanıcılara hem içerik benzerliğine hem de benzer kullanıcı zevklerine dayalı olarak kişiselleştirilmiş film önerileri sunmaktır.",
        "about_tech_stack_header": "Kullanılan Teknolojiler",
        "about_models_header": "Tavsiye Modelleri",
        "about_content_model_title": "İçerik Tabanlı Filtreleme (Content-Based Filtering)",
        "about_content_model_desc": "Bu model, filmlerin metinsel özelliklerine (özet, tür, anahtar kelimeler, yönetmen) dayanır. Birbirine benzeyen filmleri bulmak için **TF-IDF** vektörleştirmesi ve **Kosinüs Benzerliği** (Cosine Similarity) metriklerini kullanır. Temel mantığı: 'Bu filmi sevdiysen, buna benzer içeriklere sahip şu filmleri de sevebilirsin.'",
        "about_collab_model_title": "İşbirlikçi Filtreleme (Collaborative Filtering)",
        "about_collab_model_desc": "Bu model, kullanıcıların geçmişte filmlere verdiği puanları analiz eder. **SVD (Singular Value Decomposition)** algoritmasını kullanarak kullanıcılar ve filmler arasındaki gizli ilişkileri (latent factors) ortaya çıkarır. Temel mantığı: 'Seninle benzer zevklere sahip diğer kullanıcıların sevdiği filmleri sen de muhtemelen seversin.'",
        "about_hybrid_model_title": "Hibrit Model",
        "about_hybrid_model_desc": "Uygulamanın varsayılan 'Hibrit' seçeneği, yukarıdaki iki modelden gelen sonuçları birleştirerek daha zengin ve isabetli bir tavsiye listesi sunar.",
        "language_select": "Dil / Language"
    },
    "en": {
        "app_title": "🎬 Advanced Movie Recommender System",
        "app_subtitle": "Select a movie and let our hybrid model recommend the best matches for you!",
        "settings_header": "Settings",
        "select_movie_label": "Select a movie to start:",
        "num_recs_label": "Number of Recommendations",
        "rec_type_label": "Recommendation Type",
        "rec_type_hybrid": "✨ Hybrid",
        "rec_type_content": "🧠 Content-Based",
        "rec_type_collab": "👥 Collaborative Filtering",
        "get_recs_button": "Get Recommendations",
        "spinner_text": "🎬 Preparing your movies...",
        "recommendations_for": "Recommendations for:",
        "no_recs_found": "No recommendations found for this movie. Please try another one.",
        "welcome_message": "Please select a movie and click the 'Get Recommendations' button.",
        "tab_recommender": "🤖 Recommender Tool",
        "tab_about": "📖 About the Project",
        "about_header": "Project Purpose and Technical Details",
        "about_intro": "This project is a hybrid movie recommender system developed for Day 12 of the #30DaysOfAI challenge. The goal is to provide users with personalized movie suggestions based on both content similarity and similar user tastes.",
        "about_tech_stack_header": "Technology Stack",
        "about_models_header": "Recommendation Models",
        "about_content_model_title": "Content-Based Filtering",
        "about_content_model_desc": "This model relies on the textual attributes of movies (overview, genre, keywords, director). It uses **TF-IDF** vectorization and **Cosine Similarity** to find movies that are similar to each other. The core logic is: 'If you liked this movie, you might also like these other movies with similar content.'",
        "about_collab_model_title": "Collaborative Filtering",
        "about_collab_model_desc": "This model analyzes past user ratings. Using the **SVD (Singular Value Decomposition)** algorithm, it uncovers latent factors between users and movies. The core logic is: 'You will probably like movies that other users with similar tastes have liked.'",
        "about_hybrid_model_title": "Hybrid Model",
        "about_hybrid_model_desc": "The app's default 'Hybrid' option combines the results from the two models above to provide a richer and more accurate list of recommendations.",
        "language_select": "Dil / Language"
    }
}

st.set_page_config(page_title="Film Tavsiye Sistemi", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

if 'lang' not in st.session_state:
    st.session_state.lang = 'tr'

def T(key):
    return LANGUAGES[st.session_state.lang][key]

@st.cache_resource
def load_resources():
    try:
        svd_model = joblib.load('svd_model.pkl')
        cosine_sim = joblib.load('cosine_sim.pkl')
        movies_df = joblib.load('movies_df.pkl')
        return svd_model, cosine_sim, movies_df
    except FileNotFoundError:
        return None, None, None

svd_model, cosine_sim, movies_df = load_resources()

def get_content_recommendations(movie_title, cosine_sim, movies_df, num_recs=10):
    if movie_title not in movies_df['title'].values:
        return []
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recs+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['movieId'].iloc[movie_indices].tolist()

def get_collaborative_recommendations(movie_id, svd_model, num_recs=10):
    try:
        inner_id = svd_model.trainset.to_inner_iid(movie_id)
        movie_vector = svd_model.qi[inner_id]
        all_movie_vectors = svd_model.qi
        sims = np.dot(all_movie_vectors, movie_vector) / (np.linalg.norm(all_movie_vectors, axis=1) * np.linalg.norm(movie_vector))
        similar_inner_ids = np.argsort(sims)[::-1][1:num_recs+1]
        similar_movie_ids = [svd_model.trainset.to_raw_iid(i) for i in similar_inner_ids]
        return similar_movie_ids
    except (ValueError, IndexError):
        return []

st.markdown("""
<style>
    .movie-card {
        background-color: #2c2c2c;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #1DB954;
    }
    .movie-genres {
        font-size: 0.8rem;
        font-style: italic;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

st.title(T("app_title"))
st.write(T("app_subtitle"))

if svd_model is None or cosine_sim is None or movies_df is None:
    st.error("Model dosyaları bulunamadı! Lütfen önce `train_model.py` script'ini çalıştırın.")
    st.stop()

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
                if recommendation_type == T("rec_type_hybrid"):
                    content_recs = get_content_recommendations(selected_movie_title, cosine_sim, movies_df, num_recommendations)
                    collab_recs = get_collaborative_recommendations(selected_movie_id, svd_model, num_recommendations)
                    recs_set = set()
                    for rec in content_recs + collab_recs:
                        if rec not in recs_set and rec != selected_movie_id:
                            recommendations.append(rec)
                            recs_set.add(rec)
                
                elif recommendation_type == T("rec_type_content"):
                    recommendations = get_content_recommendations(selected_movie_title, cosine_sim, movies_df, num_recommendations)

                elif recommendation_type == T("rec_type_collab"):
                     recommendations = get_collaborative_recommendations(selected_movie_id, svd_model, num_recommendations)

            if not recommendations:
                st.warning(T("no_recs_found"))
            else:
                final_recs = recommendations[:num_recommendations]
                
                cols = st.columns(4) 
                for i, movie_id in enumerate(final_recs):
                    movie_details = movies_df[movies_df['movieId'] == movie_id]
                    if not movie_details.empty:
                        movie_details = movie_details.iloc[0]
                        genres_text = ', '.join(movie_details['genres_list'])
                        
                        with cols[i % 4]:
                            st.markdown(f"""
                            <div class="movie-card" title="{movie_details['overview'].replace('"',"'")}">
                                <p class="movie-title">{movie_details['title']}</p>
                                <p class="movie-genres">{genres_text}</p>
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
    - **Pandas** for data manipulation
    - **Scikit-learn** for TF-IDF vectorization
    - **Surprise** for Collaborative Filtering (SVD)
    """)

    st.subheader(T("about_models_header"))
    st.info(f"**{T('about_content_model_title')}**\n\n{T('about_content_model_desc')}")
    st.info(f"**{T('about_collab_model_title')}**\n\n{T('about_collab_model_desc')}")
    st.success(f"**{T('about_hybrid_model_title')}**\n\n{T('about_hybrid_model_desc')}")