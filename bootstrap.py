from pathlib import Path
from urllib.request import urlretrieve
import runpy

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "clean_music_dataset.csv"

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Aimans20/music_dataset/resolve/main/"
    "clean_music_dataset.csv?download=true"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    temp_file = DATA_DIR / "clean_music_dataset.csv.part"

    try:
        print("Downloading dataset from Hugging Face...")
        urlretrieve(DATASET_URL, temp_file)
        temp_file.replace(DATA_FILE)
        print("Dataset downloaded successfully.")
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
