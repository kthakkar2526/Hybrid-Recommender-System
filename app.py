import streamlit as st
from content_based_filtering import recommend_songs as content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from collaborative_filtering import recommend_songs as collaborative_recommendation
from hybrid_recommendation import HybridRecommenderSystem

# load the data
cleaned_data = "data/cleaned_data.csv"
songs_data = pd.read_csv(cleaned_data)

# load the transformed data
transformed_path = "data/transformed_data.npz"
transformed_data = load_npz(transformed_path)

# load the track ids
track_path = "data/track_ids.npy"
track_ids = load(track_path, allow_pickle=True)

# load the filtered songs data
filtered_path = "data/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_path)

# load the interaction matrix
interaction_matrix_path = "data/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Subheader
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Text Input
input_song_name = st.text_input('Enter a song name:')
st.write('You entered:', input_song_name)
# artist name
input_artist_name = st.text_input('Enter the artist name:')
st.write('You entered:', input_artist_name)
# lowercase the input
input_song_name = input_song_name.lower()
input_artist_name = input_artist_name.lower()

# k recommendations
num_recommendations = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)

# Filtering Type
filter_type = st.selectbox('Choose a filtering type:', ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Filtering'])

if filter_type == 'Collaborative Filtering':
    if st.button('Get Recommendations'):
        if ((filtered_data["name"] == input_song_name) & (filtered_data["artist"] == input_artist_name)).any():
            st.write('Recommendations for', f"**{input_song_name}** by **{input_artist_name}**")
            recommendations = collaborative_recommendation(song_title=input_song_name,
                                                          artist_name=input_artist_name,
                                                          track_ids=track_ids,
                                                          songs_df=filtered_data,
                                                          interaction_matrix=interaction_matrix,
                                                          num_recommendations=num_recommendations)
            
            # Display Recommendations
            for index, recommendation in recommendations.iterrows():
                song_title = recommendation['name'].title()
                artist_title = recommendation['artist'].title()

                if index == 0:
                    st.markdown('## Currently Playing')
                    st.markdown(f"#### **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif index == 1:
                    st.markdown('### Next Up 🎵')
                    st.markdown(f"#### **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {index}. **{song_title}** by **{artist_title}**")   
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                    
        else:
            st.write('No recommendations found for this song.')

elif filter_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):
        if ((songs_data["name"] == input_song_name) & (songs_data['artist'] == input_artist_name)).any():
            st.write('Recommendations for', f"**{input_song_name}** by **{input_artist_name}**")
            recommendations = content_recommendation(song_title=input_song_name,
                                                     songs_data=songs_data,
                                                     transformed_data=transformed_data,
                                                     k=num_recommendations)
            
            # Display Recommendations
            for index, recommendation in recommendations.iterrows():
                song_title = recommendation['name'].title()
                artist_title = recommendation['artist'].title()
                
                if index == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif index == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {index}. **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {index}. **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {input_song_name} by {input_artist_name} in our database. Please try another song.")

else:
    hybrid_recommender = HybridRecommenderSystem(num_recommendations=10, content_weight=0.5)
    if st.button('Get Recommendations'):
        if ((songs_data["name"] == input_song_name) & (songs_data['artist'] == input_artist_name)).any():
            st.write('Recommendations for', f"**{input_song_name}** by **{input_artist_name}**")
            recommendations = hybrid_recommender.give_recommendations(song_title=input_song_name,
                                                        artist_title=input_artist_name,
                                                        song_data=songs_data,
                                                        track_ids=track_ids,
                                                        transformed_matrix=transformed_data,
                                                        interaction_matrix=interaction_matrix,
                                                        k=num_recommendations)
            
            # Display Recommendations
            for index, recommendation in recommendations.iterrows():
                song_title = recommendation['name'].title()
                artist_title = recommendation['artist'].title()
                
                if index == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif index == 1:   
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {index}. **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {index}. **{song_title}** by **{artist_title}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {input_song_name} by {input_artist_name} in our database. Please try another song.")