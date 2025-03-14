import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

output_track_ids_path = "data/track_ids.npy"
output_filtered_data_path = "data/collab_filtered_data.csv"
output_interaction_matrix_path = "data/interaction_matrix.npz"
input_songs_data_path = "data/cleaned_data.csv"
input_user_history_path = "data/User Listening History.csv"


def filter_song_data(songs_df: pd.DataFrame, track_ids_list: list, save_path: str) -> pd.DataFrame:
    filtered_songs = songs_df[songs_df["track_id"].isin(track_ids_list)]
    filtered_songs.sort_values(by="track_id", inplace=True)
    filtered_songs.reset_index(drop=True, inplace=True)
    save_dataframe_to_csv(filtered_songs, save_path)
    
    return filtered_songs


def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    dataframe.to_csv(file_path, index=False)
    
    
def save_sparse_matrix_to_file(matrix: csr_matrix, file_path: str) -> None:
    save_npz(file_path, matrix)


def generate_interaction_matrix(history_df: dd.DataFrame, track_ids_path, matrix_save_path) -> csr_matrix:
    df_copy = history_df.copy()
    
    df_copy['playcount'] = df_copy['playcount'].astype(np.float64)
    df_copy = df_copy.categorize(columns=['user_id', 'track_id'])
    
    user_indices = df_copy['user_id'].cat.codes
    track_indices = df_copy['track_id'].cat.codes
    
    track_ids = df_copy['track_id'].cat.categories.values
    
    np.save(track_ids_path, track_ids, allow_pickle=True)
    
    df_copy = df_copy.assign(
        user_index=user_indices,
        track_index=track_indices
    )
    
    interaction_matrix_df = df_copy.groupby(['track_index', 'user_index'])['playcount'].sum().reset_index()
    interaction_matrix_df = interaction_matrix_df.compute()
    
    row_idx = interaction_matrix_df['track_index']
    col_idx = interaction_matrix_df['user_index']
    values = interaction_matrix_df['playcount']
    
    num_tracks = row_idx.nunique()
    num_users = col_idx.nunique()
    
    interaction_matrix = csr_matrix((values, (row_idx, col_idx)), shape=(num_tracks, num_users))
    
    save_sparse_matrix_to_file(interaction_matrix, matrix_save_path)
    
    
def recommend_songs(song_title, artist_name, track_ids, songs_df, interaction_matrix, num_recommendations=5):
    song_title = song_title.lower()
    artist_name = artist_name.lower()
    
    song_row = songs_df.loc[(songs_df["name"] == song_title) & (songs_df["artist"] == artist_name)]
    input_track_id = song_row['track_id'].values.item()
  
    index = np.where(track_ids == input_track_id)[0].item()
    input_vector = interaction_matrix[index]
    
    similarity_scores = cosine_similarity(input_vector, interaction_matrix)
    recommendation_indices = np.argsort(similarity_scores.ravel())[-num_recommendations-1:][::-1]
    
    recommended_track_ids = track_ids[recommendation_indices]
    top_scores = np.sort(similarity_scores.ravel())[-num_recommendations-1:][::-1]
    
    scores_dataframe = pd.DataFrame({"track_id": recommended_track_ids.tolist(),
                                      "score": top_scores})
    
    top_recommendations = (
        songs_df
        .loc[songs_df["track_id"].isin(recommended_track_ids)]
        .merge(scores_dataframe, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )
    
    return top_recommendations


def main():
    user_history_data = dd.read_csv(input_user_history_path)
    unique_track_ids = user_history_data.loc[:, "track_id"].unique().compute()
    unique_track_ids = unique_track_ids.tolist()
    
    songs_data = pd.read_csv(input_songs_data_path)
    filter_song_data(songs_data, unique_track_ids, output_filtered_data_path)
    
    generate_interaction_matrix(user_history_data, output_track_ids_path, output_interaction_matrix_path)


if __name__ == "__main__":
    main()