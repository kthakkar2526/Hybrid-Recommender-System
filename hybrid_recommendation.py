import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommenderSystem:
    
    def __init__(self,  
                 num_recommendations: int, 
                 content_weight: float):
        
        self.num_recommendations = num_recommendations
        self.content_weight = content_weight
        self.collaborative_weight = 1 - content_weight
        
    def __calculate_content_similarities(self, song_title, artist_title, song_data, transformed_matrix):
        row_song = song_data.loc[(song_data["name"] == song_title) & (song_data["artist"] == artist_title)]
        if row_song.empty:
            return None  # Return None if song is not found
        
        song_index = row_song.index[0]
        input_vector = transformed_matrix[song_index].reshape(1,-1)
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        return content_similarity_scores
        
    def __calculate_collaborative_similarities(self, song_title, artist_title, track_ids, song_data, interaction_matrix):
        row_song = song_data.loc[(song_data["name"] == song_title) & (song_data["artist"] == artist_title)]
        if row_song.empty:
            return None  # Return None if song is not found

        input_track_id = row_song['track_id'].values.item()
        index = np.where(track_ids == input_track_id)[0]

        if len(index) == 0:
            return None  # If song not found in track_ids, return None

        index = index[0]
        input_array = interaction_matrix[index]
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        return collaborative_similarity_scores
    
    def __normalize_scores(self, similarity_scores):
        if similarity_scores is None:
            return None  # Skip normalization if scores are None
        
        min_score = np.min(similarity_scores)
        max_score = np.max(similarity_scores)
        if max_score - min_score == 0:
            return similarity_scores  # Avoid division by zero
        
        normalized_scores = (similarity_scores - min_score) / (max_score - min_score)
        return normalized_scores
    
    def __combine_weights(self, content_scores, collaborative_scores):
        if content_scores is None or collaborative_scores is None:
            return None  # Skip combination if any of the scores are missing
        
        min_length = min(content_scores.shape[1], collaborative_scores.shape[1])
        content_scores = content_scores[:, :min_length]
        collaborative_scores = collaborative_scores[:, :min_length]

        combined_scores = (self.content_weight * content_scores) + (self.collaborative_weight * collaborative_scores)
        return combined_scores
    
    def give_recommendations(self, song_title, artist_title, song_data, track_ids, transformed_matrix, interaction_matrix, k=10):
        content_similarities = self.__calculate_content_similarities(song_title, artist_title, song_data, transformed_matrix)
        collaborative_similarities = self.__calculate_collaborative_similarities(song_title, artist_title, track_ids, song_data, interaction_matrix)

        # If any similarity score is None, return an empty DataFrame
        if content_similarities is None or collaborative_similarities is None:
            return pd.DataFrame(columns=["name", "artist", "spotify_preview_url"])

        normalized_content_similarities = self.__normalize_scores(content_similarities)
        normalized_collaborative_similarities = self.__normalize_scores(collaborative_similarities)

        combined_scores = self.__combine_weights(normalized_content_similarities, normalized_collaborative_similarities)

        # If combination failed, return empty recommendations
        if combined_scores is None:
            return pd.DataFrame(columns=["name", "artist", "spotify_preview_url"])

        recommendation_indices = np.argsort(combined_scores.ravel())[-k-1:][::-1] 
        
        recommendation_track_ids = track_ids[recommendation_indices]
       
        top_scores = np.sort(combined_scores.ravel())[-k-1:][::-1]
        
        scores_df = pd.DataFrame({"track_id": recommendation_track_ids.tolist(),
                                  "score": top_scores})
        top_songs = (
                        song_data
                        .loc[song_data["track_id"].isin(recommendation_track_ids)]
                        .merge(scores_df, on="track_id")
                        .sort_values(by="score", ascending=False)
                        .drop(columns=["track_id", "score"])
                        .reset_index(drop=True)
                    )
        
        return top_songs