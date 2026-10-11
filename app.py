import requests
from urllib.parse import quote_plus

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #090d18 0%, #111827 55%, #17112b 100%);
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #ffffff !important;
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #e9e7ff !important;
    }

    p, label {
        color: #cbd5e1;
    }

    div[data-testid="stTextInput"] input {
        background: #171e30;
        color: white;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 0.75rem;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: #171e30;
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        background: linear-gradient(90deg, #7c3aed, #db2777);
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        border: 1px solid #e9d5ff;
        box-shadow: 0 0 20px #7c3aed55;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #30384c;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #30384c;
    }
</style>
""", unsafe_allow_html=True)

DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "clean_music_dataset.csv"
)

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

@st.cache_resource(show_spinner="Preparing the recommendation engine...")
def load_engine(file_path, file_mtime):
    df = pd.read_csv(file_path, low_memory=False)

    required = ["name", "artists"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    features = [c for c in FEATURES if c in df.columns]
    if not features:
        raise ValueError("No supported audio feature columns were found.")

    df = df.reset_index(drop=True)
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["artists"] = df["artists"].fillna("").astype(str).str.strip()

    X = df[features].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    model = NearestNeighbors(
    metric="euclidean",
    algorithm="brute",
    n_jobs=-1,
)
    model.fit(X_scaled)

    return df, features, X_scaled, model


st.markdown(
    "<div style='color:#c084fc;font-weight:700;"
    "letter-spacing:3px;font-size:12px;'>MUSIC • DISCOVER • REPEAT</div>",
    unsafe_allow_html=True,
)

@st.cache_data(ttl=86400, show_spinner=False)
def get_album_art(song, artist):
    query = f"{song} {artist}"

    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "entity": "song",
                "limit": 5,
            },
            timeout=5,
        )
        response.raise_for_status()
        tracks = response.json().get("results", [])

        if tracks:
            return tracks[0].get("artworkUrl100", "")

    except Exception:
        pass

    return ""

st.title("Find Your Next Favorite Song")
st.write(
    "Explore music that matches your sound. Search for a track, "
    "select the version you want, and discover similar songs."
)

st.divider()

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info(
        "Place clean_music_dataset.csv inside your project's data folder."
    )
    st.stop()

try:
    df, features, X_scaled, model = load_engine(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()


query = st.text_input("Search for a song or artist", placeholder="Enter a song title")

if query:
    matches = df[
        df["name"].str.contains(query, case=False, regex=False, na=False)
    ]

    if matches.empty:
        st.warning("No matching songs found. Try another title.")
    else:
        choices = [
            f"{idx} | {row['name']} — {row['artists']}"
            for idx, row in matches.head(100).iterrows()
        ]

        selected = st.selectbox("Choose the exact track", choices)
        selected_idx = int(selected.split(" | ", 1)[0])
        top_n = st.slider("Number of recommendations", 5, 20, 10)


        if st.button("Recommend similar songs"):
            distances, indices = model.kneighbors(
                X_scaled[selected_idx].reshape(1, -1),
                n_neighbors=min(top_n + 20, len(df)),
            )

            results = []

            for distance, idx in zip(distances[0], indices[0]):
                if idx == selected_idx:
                    continue

                row = df.iloc[idx]

                results.append({
                    "Song": row["name"],
                    "Artist": row["artists"],
                })

                if len(results) >= top_n:
                    break


            if results:
                st.subheader(
                    f"Songs similar to {df.iloc[selected_idx]['name']}"
                )

                for song in results:
                    title = song["Song"]
                    artist = song["Artist"]

                    artwork = get_album_art(title, artist)

                    spotify_search = f"{title} {artist}"
                    spotify_url = (
                        "https://open.spotify.com/search/"
                        + quote_plus(spotify_search)
                    )
                    spotify_app_url = (
                        "spotify:search:" + quote_plus(spotify_search)
                    )

                    youtube_url = (
                        "https://www.youtube.com/results?search_query="
                        + quote_plus(f"{title} {artist}")
                    )

                    with st.container(border=True):
                        image_col, details_col = st.columns(
                            [1, 4],
                            vertical_alignment="center",
                        )

                        with image_col:
                            if artwork:
                                st.image(artwork, width=110)
                            else:
                                st.markdown("🎵")

                        with details_col:
                            st.markdown(f"### {title}")
                            st.write(artist)

                            link_col1, link_col2 = st.columns(2)

                            with link_col1:
                                st.link_button(
                                    "▶ Open Spotify",
                                    spotify_url,
                                    use_container_width=True,
                                )

                                st.markdown(
                                    f"[Try Spotify app directly]({spotify_app_url})"
                                )

                            with link_col2:
                                st.link_button(
                                    "▶ Open YouTube",
                                    youtube_url,
                                    use_container_width=True,
                                )

            else:
                st.warning("No other recommendations were found.")
