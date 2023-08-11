import streamlit as st
import pandas as pd
import numpy as np
from src.models import BaselinePredictor as BP
from src.models import projectLib as lib
from src.models import RBMmodel as RBM

training_data = lib.getTrainingData()
unique_movie_ids = np.unique(training_data[:, 0])  # Assuming Movie_ID is the first column
movie_titles = ["Movie " + str(int(movie_id)) for movie_id in unique_movie_ids]

# Function to get top recommendations
def get_top_recommendations(user_ratings, model_name):
    # Load the training data
    training_data = lib.getTrainingData()
    validation_data = lib.getValidationData()

    trStats = lib.getUsefulStats(training_data)
    allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training_data)

    if model_name == "RBM":
        model = RBM.RBMRecommender(F=20, epochs=100, learning_rate=0.02)
    else:
        # Replace with your BaselinePredictor class
        model = BP.Model()

    model.train(training_data, validation_data)
    predicted_ratings = model.predict_for_users(trStats, allUsersRatings)

    # Get top 3 movie indices
    top_indices = np.argsort(predicted_ratings)[-3:]
    
    # Convert to a flat list of integers
    top_indices = [int(i) for i in top_indices.flatten()]

    # Return the top movie titles
    return [movie_titles[i] for i in top_indices]

# Streamlit app
def main():

    st.title("🎬 Movie Recommender App 🍿")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")
    model = st.sidebar.radio("Select Model", ["RBM", "Baseline Predictor"])

    st.markdown("## 🎥 Rate Movies")
    user_ratings = {}
    
    # Select a random subset of 20 movies for the user to rate
    subset_movies = np.random.choice(movie_titles, 20, replace=False)
    for movie in subset_movies:
        rating = st.slider(f"Rate {movie} (0 = not seen)", 0, 5, 0)
        if rating > 0:
            user_ratings[movie] = rating

    if st.button("Get Recommendations"):
        # Get recommendations based on user ratings
        recommended_movies = get_top_recommendations(user_ratings, model)

        # Display recommendations
        st.markdown(f"## 🌟 Top Movie Recommendations using {model}")
        for movie in recommended_movies:
            st.write(f"🎬 {movie}")

    st.sidebar.text("Note: Rate some movies and get personalized recommendations!")
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2023 MovieRecommender")


if __name__ == "__main__":
    main()
