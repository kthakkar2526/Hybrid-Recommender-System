import streamlit as st
from content_based_filtering import recommend_songs as content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from collaborative_filtering import recommend_songs as collaborative_recommendation
from hybrid_recommendation import HybridRecommenderSystem

# Load the data
cleaned_data_path = "data/cleaned_data.csv"
songs_data = pd.read_csv(cleaned_data_path, encoding="utf-8")

# Load the transformed data
transformed_data_path = "data/transformed_data.npz"
transformed_data = load_npz(transformed_data_path)

# Load the track ids
track_ids_path = "data/track_ids.npy"
track_ids = load(track_ids_path, allow_pickle=True)

# Load the filtered songs data
filtered_data_path = "data/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_data_path)

# Load the interaction matrix
interaction_matrix_path = "data/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and select a recommendation method:')

# User Input
song_title = st.text_input('Enter Song Title:')
artist_name = st.text_input('Enter Artist Name:')

# Recommendation Type Selection
rec_type = st.radio(
    "Choose a recommendation type:",
    ('Content-Based', 'Collaborative Filtering', 'Hybrid'))

# Number of recommendations selection
num_recommendations = st.slider('Number of recommendations:', 1, 20, 10)

if st.button('Get Recommendations') and song_title and artist_name:
    if rec_type == 'Content-Based':
        recommendations = content_recommendation(song_title, artist_name, songs_data, transformed_data)
    elif rec_type == 'Collaborative Filtering':
    elif rec_type == 'Hybrid':
        hybrid_recommender = HybridRecommenderSystem(num_recommendations=10, content_weight=0.5)  # Default values
        recommendations = hybrid_recommender.get_hybrid_recommendations(song_title, artist_name, songs_data, transformed_data, track_ids, filtered_data, interaction_matrix)

        recommendations = collaborative_recommendation(song_title, artist_name, track_ids, filtered_data, interaction_matrix)
    elif rec_type == 'Hybrid':
        hybrid_recommender = HybridRecommenderSystem(num_recommendations, content_weight=0.5)  # Default weight
        recommendations = hybrid_recommender.get_hybrid_recommendations(song_title, artist_name, songs_data, transformed_data, track_ids, filtered_data, interaction_matrix)
    
    if recommendations:
        st.write('### Recommended Songs:')
        for song, artist in recommendations:
            st.write(f'- {song} by {artist}')
    else:
        st.write('No recommendations found. Please try a different song.')