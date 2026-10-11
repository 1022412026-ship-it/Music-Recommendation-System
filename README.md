# Music Recommendation System

## Project Overview
An AI-based music recommendation prototype developed using Python and Streamlit. It recommends songs with similar audio characteristics and displays song artwork with Spotify and YouTube search links.

## Objectives
- Search for songs in a music dataset.
- Analyse numerical audio features.
- Recommend similar tracks using a nearest-neighbours model.
- Display recommendations in a user-friendly web interface.
- Provide links to search for tracks on Spotify and YouTube.

## Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Requests
- iTunes Search API for album artwork lookup

## Dataset
The working dataset contains approximately 1.14 million tracks.

File expected by the application:
`data/clean_music_dataset.csv`

Dataset source: ADD YOUR ORIGINAL DATASET LINK HERE.

The dataset is not included in this repository because of its large file size. Download it from the original source and place the cleaned CSV in the `data` folder.

## How to Run

1. Install Python.
2. Open the project folder in a terminal.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Place `clean_music_dataset.csv` inside the `data` folder.
5. Start the application:

   `python -m streamlit run app.py`

6. Open the local URL displayed in the terminal.

## How It Works
1. Load the music dataset.
2. Select available audio features.
3. Handle missing feature values and scale the features.
4. Fit a nearest-neighbours recommendation model.
5. Search for a song and select a matching track.
6. Display recommendations with artwork and external search links.

## Current Features
- Song search and selection
- Audio-feature-based recommendations
- Album artwork lookup
- Spotify and YouTube links
- Responsive Streamlit interface

## Limitations and Future Work
- Verify recommendation quality using a formal evaluation.
- Improve exact song and artwork matching.
- Explore official streaming-service API integration.
- Investigate automatic dataset updates.

## Developer
Individual college project.
