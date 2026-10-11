
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# MUSIC RECOMMENDATION SYSTEM
# ==========================================

print("Loading music dataset...")

DATASET_PATH = Path(
    r"C:\Users\ADMIN\OneDrive\Desktop\Music Recommendation System\data\clean_music_dataset.csv"
)

df = pd.read_csv("C:/Users/ADMIN/OneDrive/Desktop/Music Recommendation System/data/clean_music_dataset.csv", low_memory=False)
print(f"Dataset loaded: {len(df):,} songs")

# ==========================================
# STEP 1: CLEAN THE DATA
# ==========================================

required_columns = [
    "id", "name", "artists",
    "danceability", "energy", "loudness",
    "speechiness", "acousticness",
    "instrumentalness", "liveness",
    "valence", "tempo"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Remove rows without song titles or artists
df = df.dropna(subset=["name", "artists"]).copy()

# Convert audio features to numeric values
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing or infinite feature values
df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

df = df.dropna(subset=features).reset_index(drop=True)

print(f"Songs after cleaning: {len(df):,}")

# ==========================================
# STEP 2: SCALE AUDIO FEATURES
# ==========================================

print("Preparing audio features...")

scaler = StandardScaler()
X = scaler.fit_transform(df[features]).astype(np.float32)

# Normalize each song's feature vector for cosine similarity
from sklearn.preprocessing import normalize

X = normalize(X, norm="l2")

print("Audio features prepared!")

# ==========================================
# STEP 3: BUILD THE RECOMMENDATION MODEL
# ==========================================

print("Building recommendation model...")

model = NearestNeighbors(
    n_neighbors=11,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

model.fit(X)

print("Recommendation model created successfully!")

# ==========================================
# STEP 4: RECOMMEND SONGS
# ==========================================

def recommend_songs(song_name, top_n=10, artist_name=None):
    """
    Recommend songs using audio-feature similarity.
    An optional artist name disambiguates duplicate song titles.
    """

    matches = df[
        df["name"].str.casefold() == song_name.strip().casefold()
    ]

    if artist_name:
        matches = matches[
            matches["artists"].str.contains(
                artist_name, case=False, regex=False, na=False
            )
        ]

    if matches.empty:
        print(f"\nSong not found: {song_name}")
        print("Check the spelling or try part of the title.")
        return pd.DataFrame()

    if len(matches) > 1:
        print(f"\nFound {len(matches)} matching tracks.")
        print("Using the first matching track. Specify artist_name")
        print("if you want to select a particular artist.")

    # Select the requested track
    song_idx = int(matches.index[0])

    # Ask the model for nearest neighbors
    distances, indices = model.kneighbors(
        X[song_idx:song_idx + 1],
        n_neighbors=min(top_n + 1, len(df))
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        # Exclude the exact query row
        if idx == song_idx:
            continue

        results.append({
            "song": df.iloc[idx]["name"],
            "artist": df.iloc[idx]["artists"],
            "album": df.iloc[idx]["album"],
            "similarity_score": round(
                max(0.0, 1.0 - float(distance)), 4
            )
        })

        if len(results) >= top_n:
            break

    return pd.DataFrame(results)


# ==========================================
# STEP 5: TEST THE SYSTEM
# ==========================================

if __name__ == "__main__":
    test_song = "Testify"
    test_artist = "Rage Against The Machine"

    recommendations = recommend_songs(
        song_name=test_song,
        artist_name=test_artist,
        top_n=10
    )

    print(f"\nRecommendations for: {test_song}")
    print("-" * 75)

    if not recommendations.empty:
        print(recommendations.to_string(index=False))

        # Save test results
        output_path = Path(__file__).parent / "test_recommendations.csv"
        recommendations.to_csv(output_path, index=False)

        print(f"\nResults saved to: {output_path}")
    else:
        print("No recommendations generated.")

print("\n--- TESTIFY FEATURE CHECK ---")

matches = df
[
    (df["name"].str.casefold() == "testify")
    & df["artists"].str.contains(
        "Rage Against The Machine",
        case=False,
        regex=False,
        na=False
    )
]

if matches.empty:
    print("Testify by Rage Against The Machine was not found.")
else:
    print(matches[["name", "artists"] + features].to_string(index=False))

    
print("\n--- DEBUG: TESTIFY SEARCH ---")

print("Exact title matches:")
print(
    df.loc[
        df["name"].astype(str).str.strip().str.casefold() == "testify",
        ["id", "name", "artists"]
    ].to_string(index=False)
)

print("\nTitles containing Testify:")
print(
    df.loc[
        df["name"].astype(str).str.contains(
            "testify", case=False, na=False, regex=False
        ),
        ["id", "name", "artists"]
    ].head(20).to_string(index=False)
)

print("\nTotal matching titles:",
      df["name"].astype(str).str.strip().str.casefold().eq("testify").sum())


print("\n--- DEBUG: TESTIFY SEARCH ---")

print("Dataset rows:", len(df))
print("Testify matches:", df["name"].astype(str).str.strip().str.casefold().eq("testify").sum())

print(
    df.loc[
        df["name"].astype(str).str.strip().str.casefold().eq("testify"),
        ["id", "name", "artists"]
    ].to_string(index=False)
)

print("--- END DEBUG ---")
