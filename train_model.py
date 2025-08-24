import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import warnings
import json
import ast

warnings.filterwarnings("ignore")

DATA_PATH = "data/"

def train_svd_model():
    """
    Trains the Collaborative Filtering SVD model and saves it.
    """
    print("--- SVD Model Training Started ---")
    
    ratings = pd.read_csv(DATA_PATH + "ratings_small.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    model = SVD(n_factors=150, n_epochs=20, lr_all=0.005, reg_all=0.04, random_state=42)
    model.fit(trainset)
    
    joblib.dump(model, 'svd_model.pkl')
    print("--- SVD Model Training Completed ---")

def create_content_model():
    """
    Creates and saves artifacts for the Content-Based Filtering model.
    This includes a cosine similarity matrix and a consolidated movie dataframe.
    """
    print("--- Content-Based Model Creation Started ---")

    meta = pd.read_csv(DATA_PATH + 'movies_metadata.csv', low_memory=False)
    credits = pd.read_csv(DATA_PATH + 'credits.csv')
    keywords = pd.read_csv(DATA_PATH + 'keywords.csv')
    links = pd.read_csv(DATA_PATH + 'links_small.csv')

    meta['id'] = pd.to_numeric(meta['id'], errors='coerce')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')

    meta.dropna(subset=['id'], inplace=True)
    meta['id'] = meta['id'].astype(int)

    df = meta.merge(credits, on='id').merge(keywords, on='id')
    
    links = links[links['tmdbId'].notna()]
    links['tmdbId'] = links['tmdbId'].astype(int)
    df = df[df['id'].isin(links['tmdbId'])]

    def parse_literal(data):
        try:
            return ast.literal_eval(data)
        except (ValueError, SyntaxError):
            return []

    df['genres_list'] = df['genres'].apply(parse_literal).apply(lambda x: [i['name'] for i in x])
    df['genres'] = df['genres_list'].apply(lambda x: ' '.join(map(str, x)))

    df['keywords_list'] = df['keywords'].apply(parse_literal).apply(lambda x: [i['name'] for i in x])
    df['keywords'] = df['keywords_list'].apply(lambda x: ' '.join(map(str, x)))

    def get_director(data):
        crew_list = parse_literal(data)
        for crew_member in crew_list:
            if crew_member.get('job') == 'Director':
                return crew_member.get('name', '')
        return ""
    
    df['director'] = df['crew'].apply(get_director)

    df['soup'] = df['overview'].fillna('') + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['director']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    joblib.dump(cosine_sim, 'cosine_sim.pkl')
    
    df_processed = df[['id', 'title', 'genres_list', 'overview']].rename(columns={'id': 'tmdbId'})
    final_df = df_processed.merge(links, on='tmdbId')[['movieId', 'tmdbId', 'title', 'genres_list', 'overview']]
    joblib.dump(final_df, 'movies_df.pkl')

    print("--- Content-Based Model Creation Completed ---")

if __name__ == "__main__":
    print("--- Full Training Pipeline Initiated ---")
    train_svd_model()
    create_content_model()
    print("--- Full Training Pipeline Completed Successfully! ---")