import streamlit as st
import pandas as pd
import torch
from agents.model import QNetwork

# Load ratings data (which contains both movieId and title)
ratings_df = pd.read_csv("./data/processed/data.csv")

# Get unique movie IDs based on ratings data
unique_movie_ids = ratings_df["movieId"].unique()
NUM_MOVIES = 9742  # The number of unique movies

# Get the top 50 most rated movies
most_rated = ratings_df.groupby("movieId").size().reset_index(name="count")
most_rated = most_rated.sort_values("count", ascending=False)
top_movies = most_rated.head(50)

# Merge to get movie titles
top_movies = pd.merge(
    top_movies, ratings_df[["movieId", "title"]].drop_duplicates(), on="movieId"
)


# Load the trained model
def load_model():
    model = QNetwork(5, NUM_MOVIES)
    model.load_state_dict(
        torch.load("models/dqn_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


model = load_model()

# Initialize Streamlit session
st.title("🎬 DQN-based Movie Recommender")

if "history" not in st.session_state:
    st.session_state.history = []

# UI for selecting favorite movies
st.subheader("Pick 5 movies you like 👇")

options = top_movies["title"].tolist()
selected = st.multiselect("Top Rated Movies:", options)

# Update state history
for title in selected:
    movie_id = top_movies[top_movies["title"] == title]["movieId"].values[0]
    if movie_id not in st.session_state.history:
        st.session_state.history.append(movie_id)

# Keep only last 5
st.session_state.history = st.session_state.history[-5:]

# Show selected movies
if st.session_state.history:
    st.write("🧠 Your Picks (Last 5):")
    titles = [
        top_movies[top_movies["movieId"] == mid]["title"].values[0]
        for mid in st.session_state.history
        if mid in top_movies["movieId"].values
    ]
    st.write(titles)

# Predict and recommend
if len(st.session_state.history) == 5:
    # Convert movie IDs to indices for the model
    try:
        state_tensor = torch.FloatTensor(st.session_state.history).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)
            recommended_indices = (
                torch.topk(q_values, 5, dim=1).indices.squeeze(0).tolist()
            )

        st.subheader("🎯 Recommended Movies")
        for mid in recommended_indices:
            title = ratings_df[ratings_df["movieId"] == mid]["title"].values
            if len(title) > 0:
                st.write(f"👉 {title[0]}")
            else:
                st.write(f"Movie ID {mid} (metadata missing)")
    except KeyError as e:
        st.error(f"Selected movie ID not found in model's training data: {e}")
else:
    st.warning(
        f"Please select at least {5 - len(st.session_state.history)} more movie(s) to get recommendations!"
    )
