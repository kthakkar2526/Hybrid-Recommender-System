import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import prepare_data_for_filtering, process_data
from scipy.sparse import save_npz

CLEANED_DATA_PATH = "data/cleaned_data.csv"

frequency_encode_columns = ['year']
one_hot_encode_columns = ['artist', "time_signature", "key"]
tfidf_column = 'tags'
standard_scale_columns = ["duration_ms", "loudness", "tempo"]
min_max_scale_columns = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]

def train_data_transformer(data):
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True, return_df=True), frequency_encode_columns),
        ("one_hot_encode", OneHotEncoder(handle_unknown="ignore"), one_hot_encode_columns),
        ("tfidf_vectorize", TfidfVectorizer(max_features=85), tfidf_column),
        ("standard_scale", StandardScaler(), standard_scale_columns),
        ("min_max_scale", MinMaxScaler(), min_max_scale_columns)
    ], remainder='passthrough', n_jobs=-1, force_int_remainder_cols=False)

    transformer.fit(data)
    joblib.dump(transformer, "transformer.joblib")

def transform_input_data(data):
    transformer = joblib.load("transformer.joblib")
    transformed_data = transformer.transform(data)
    return transformed_data

def save_transformed_input_data(transformed_data, save_path):
    save_npz(save_path, transformed_data)

def compute_similarity_scores(input_vector, data):
    similarity_scores = cosine_similarity(input_vector, data)
    return similarity_scores

def recommend_songs(song_title, songs_data, transformed_data, k=10):
    song_title = song_title.lower()
    song_row = songs_data.loc[(songs_data["name"] == song_title)]
    song_index = song_row.index[0]
    input_vector = transformed_data[song_index].reshape(1, -1)
    similarity_scores = compute_similarity_scores(input_vector, transformed_data)
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    top_k_list = top_k_songs_names[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)
    return top_k_list

def main(data_path):
    data = pd.read_csv(data_path)
    data_content_filtering = prepare_data_for_filtering(data)
    train_data_transformer(data_content_filtering)
    transformed_data = transform_input_data(data_content_filtering)
    save_transformed_input_data(transformed_data, "data/transformed_data.npz")

if __name__ == "__main__":
    main(CLEANED_DATA_PATH)