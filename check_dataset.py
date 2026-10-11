
from pathlib import Path
import pandas as pd

DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "clean_music_dataset.csv"
)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place clean_music_dataset.csv inside the data folder."
    )

print("Loading dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)

print("\nDataset loaded successfully!")
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nMissing values in important columns:")
for col in ["name", "artists", "danceability", "energy", "valence", "tempo"]:
    if col in df.columns:
        print(f"{col}: {df[col].isna().sum():,} missing")
    else:
        print(f"{col}: COLUMN NOT FOUND")
