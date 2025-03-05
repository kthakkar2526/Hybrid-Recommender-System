# import pandas as pd
# import dask.dataframe as dd
# from scipy.sparse import csr_matrix, save_npz
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # set paths
# # output paths
# track_ids_save_path = "data/track_ids.npy"
# filtered_data_save_path = "data/collab_filtered_data.csv"
# interaction_matrix_save_path = "data/interaction_matrix.npz"
# # input paths
# songs_data_path = "data/cleaned_data.csv"
# user_listening_history_data_path = "data/User Listening History.csv"


# def filter_songs_data(songs_data: pd.DataFrame, track_ids: list, save_df_path: str) -> pd.DataFrame:
#     """
#     Filter the songs data for the given track ids
#     """
#     # filter data based on track_ids
#     filtered_data = songs_data[songs_data["track_id"].isin(track_ids)]
#     # sort the data by track id
#     filtered_data.sort_values(by="track_id", inplace=True)
#     # rest index
#     filtered_data.reset_index(drop=True, inplace=True)
#     # save the data
#     save_pandas_data_to_csv(filtered_data, save_df_path)
    
#     return filtered_data


# def save_pandas_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
#     """
#     Save the data to a csv file
#     """
#     data.to_csv(file_path, index=False)
    
    
# def save_sparse_matrix(matrix: csr_matrix, file_path: str) -> None:
#     """
#     Save the sparse matrix to a npz file
#     """
#     save_npz(file_path, matrix)


# def create_interaction_matrix(history_data:dd.DataFrame, track_ids_save_path, save_matrix_path) -> csr_matrix:
#     # make a copy of data
#     df = history_data.copy()
    
#     # convert the playcount column to float
#     df['playcount'] = df['playcount'].astype(np.float64)
    
#     # convert string column to categorical
#     df = df.categorize(columns=['user_id', 'track_id'])
    
#     # Convert user_id and track_id to numeric indices
#     user_mapping = df['user_id'].cat.codes
#     track_mapping = df['track_id'].cat.codes
    
#     # get the list of track_ids
#     track_ids = df['track_id'].cat.categories.values
    
#     # save the categories
#     np.save(track_ids_save_path, track_ids, allow_pickle=True)
    
#     # add the index columns to the dataframe
#     df = df.assign(
#         user_idx=user_mapping,
#         track_idx=track_mapping
#     )
    
#     # create the interaction matrix
#     interaction_matrix = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
    
#     # compute the matrix
#     interaction_matrix = interaction_matrix.compute()
    
#     # get the indices to form sparse matrix
#     row_indices = interaction_matrix['track_idx']
#     col_indices = interaction_matrix['user_idx']
#     values = interaction_matrix['playcount']
    
#     # get the shape of sparse matrix
#     n_tracks = row_indices.nunique()
#     n_users = col_indices.nunique()
    
#     # create the sparse matrix
#     interaction_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
    
#     # save the sparse matrix
#     save_sparse_matrix(interaction_matrix, save_matrix_path)
    
    
# def collaborative_recommendation(song_name,artist_name,track_ids,songs_data,interaction_matrix,k=5):
#     # lowercase the song name
#     song_name = song_name.lower()
    
#     # lowercase the artist name
#     artist_name = artist_name.lower()
    
#     # fetch the row from songs data
#     song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
   
#     # track_id of input song
#     input_track_id = song_row['track_id'].values.item()
  
#     # index value of track_id
#     ind = np.where(track_ids == input_track_id)[0].item()
    
#     # fetch the input vector
#     input_array = interaction_matrix[ind]
    
#     # get similarity scores
#     similarity_scores = cosine_similarity(input_array, interaction_matrix)
    
#     # index values of recommendations
#     recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    
#     # get top k recommendations
#     recommendation_track_ids = track_ids[recommendation_indices]
    
#     # get top scores
#     top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
#     # get the songs from data and print
#     scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
#                             "score":top_scores})
    
#     top_k_songs = (
#                     songs_data
#                     .loc[songs_data["track_id"].isin(recommendation_track_ids)]
#                     .merge(scores_df,on="track_id")
#                     .sort_values(by="score",ascending=False)
#                     .drop(columns=["track_id","score"])
#                     .reset_index(drop=True)
#                     )
    
#     return top_k_songs


# def main():
#     # load the history data
#     user_data = dd.read_csv(user_listening_history_data_path)
    
#     # get the unique track ids
#     unique_track_ids = user_data.loc[:,"track_id"].unique().compute()
#     unique_track_ids = unique_track_ids.tolist()
    
#     # filter the songs data
#     songs_data = pd.read_csv(songs_data_path)
#     filter_songs_data(songs_data, unique_track_ids, filtered_data_save_path)
    
#     # create the interaction matrix
#     create_interaction_matrix(user_data, track_ids_save_path, interaction_matrix_save_path)


# if __name__ == "__main__":
#     main()


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------
# Load and Preprocess Data
# ------------------------
# Load user-item interaction data
df = pd.read_csv("data/User Listening History.csv")

# Convert categorical user and item IDs to numerical indices
df["user_id"] = df["user_id"].astype("category").cat.codes
df["track_id"] = df["track_id"].astype("category").cat.codes

num_users = df["user_id"].nunique()
num_items = df["track_id"].nunique()

# Convert to NumPy arrays
user_ids = df["user_id"].values
item_ids = df["track_id"].values
labels = np.ones(len(df))  # Implicit feedback (1 = interaction)

# -------------------
# Define Dataset Class
# -------------------
class InteractionDataset(Dataset):
    def __init__(self, users, items, labels):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# Create dataset and dataloader
dataset = InteractionDataset(user_ids, item_ids, labels)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# -------------------
# Define NCF Model
# -------------------
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32]):
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)

        # Concatenate user and item embeddings
        x = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through MLP
        x = self.mlp(x)

        # Output prediction
        x = self.output(x)
        return self.sigmoid(x)

# -------------------
# Device Configuration (Enable MPS for Mac)
# -------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = NeuralCollaborativeFiltering(num_users, num_items).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------
# Train NCF Model
# -------------------
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for users, items, labels in dataloader:
        users, items, labels = users.to(device), items.to(device), labels.to(device)

        # Forward pass
        preds = model(users, items).squeeze()
        loss = criterion(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# -------------------
# Find Similar Songs
# -------------------
def find_similar_songs(input_song_id, top_k=10):
    """
    Find similar songs to the given song ID using learned embeddings.
    
    Args:
        input_song_id (int): The track ID for which we want recommendations.
        top_k (int): Number of similar songs to return.
    
    Returns:
        DataFrame: Top recommended similar track IDs.
    """
    input_song_id = torch.LongTensor([input_song_id] * num_items).to(device)
    item_ids = torch.LongTensor(np.arange(num_items)).to(device)

    with torch.no_grad():
        scores = model(input_song_id, item_ids).squeeze()

    top_items = torch.argsort(scores, descending=True)[:top_k]
    return df[df["track_id"].isin(top_items.cpu().numpy())][["track_id"]]

# Example: Find songs similar to track 10
print(find_similar_songs(input_song_id=10))
