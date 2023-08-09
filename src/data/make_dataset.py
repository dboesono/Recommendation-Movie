# -*- coding: utf-8 -*-
import os
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define the path where you want to save the dataset
    dataset_path = 'data/raw'

    # Create the directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)

    # Download the dataset
    api.dataset_download_files('rishitjavia/netflix-movie-rating-dataset', path=dataset_path, unzip=True)

    print("Dataset downloaded successfully!")
    

# Run python script
if __name__ == "__main__":
    main()

