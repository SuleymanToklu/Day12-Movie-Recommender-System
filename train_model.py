import pandas as pd
import joblib
import warnings
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

DATA_PATH = "data/"

def create_models():
    print("--- Model Training & Processing Started ---")

    # --- Content-Based Model ---
    # Bu kısmı daha önceki çalışan versiyondan alıyoruz, ama surprise bağımlılığı yok.
    # scikit-learn kullandığı için bu kısım zaten uyumlu.
    print("1/2 - Creating Content-Based Model Artifacts...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import ast

    meta = pd.read_csv(DATA_PATH + 'movies_metadata.csv', low_memory=False)
    keywords = pd.read_csv(DATA_PATH + 'keywords.csv')
    links = pd.read_csv(DATA_PATH + 'links_small.csv')

    meta['id'] = pd.to_numeric(meta['id'], errors='coerce')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
    meta.dropna(subset=['id'], inplace=True)
    meta['id'] = meta['id'].astype(int)
    
    df = meta.merge(keywords, on='id')
    links = links[links['tmdbId'].notna()]
    links['tmdbId'] = links['tmdbId'].astype(int)
    df = df[df['id'].isin(links['tmdbId'])]

    def parse_literal(data):
        try: return ast.literal_eval(data)
        except: return []

    df['genres'] = df['genres'].apply(parse_literal).apply(lambda x: ' '.join([i['name'] for i in x]))
    df['keywords'] = df['keywords'].apply(parse_literal).apply(lambda x: ' '.join([i['name'] for i in x]))

    df['soup'] = df['overview'].fillna('') + ' ' + df['genres'] + ' ' + df['keywords']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    joblib.dump(cosine_sim, 'cosine_sim.pkl')

    df_processed = df[['id', 'title']].rename(columns={'id': 'tmdbId'})
    final_df = df_processed.merge(links, on='tmdbId')[['movieId', 'tmdbId', 'title']]
    joblib.dump(final_df, 'movies_df.pkl')
    print("Content-Based Model Artifacts Created.")

    # --- Collaborative Filtering Model (scikit-learn SVD) ---
    print("2/2 - Training scikit-learn SVD Model...")
    ratings = pd.read_csv(DATA_PATH + "ratings_small.csv")
    
    # Create a user-item matrix
    user_movie_matrix = ratings.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    # Convert to sparse matrix for efficiency
    user_movie_sparse_matrix = csr_matrix(user_movie_matrix.values)

    # Train SVD model
    svd_model = TruncatedSVD(n_components=150, random_state=42)
    svd_model.fit(user_movie_sparse_matrix)

    # Save the model and the matrix columns (for mapping movieIds)
    joblib.dump(svd_model, 'svd_model.pkl')
    joblib.dump(user_movie_matrix.columns, 'movie_ids_map.pkl')
    print("scikit-learn SVD Model Trained.")
    
    print("--- All Models Created Successfully! ---")

if __name__ == "__main__":
    create_models()