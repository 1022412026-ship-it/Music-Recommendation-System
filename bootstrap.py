from pathlib import Path
from urllib.request import urlopen
import shutil
import runpy

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "clean_music_dataset.csv"

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Aimans20/music_dataset/resolve/main/"
    "clean_music_dataset.csv"
    "?download=true"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.is_file():
    print("Downloading the music dataset from Hugging Face...")

    try:
        with urlopen(DATASET_URL, timeout=120) as response:
            with DATA_FILE.open("wb") as output:
                shutil.copyfileobj(response, output)

        if DATA_FILE.stat().st_size == 0:
            DATA_FILE.unlink(missing_ok=True)
            raise RuntimeError("The downloaded dataset file is empty.")

        print(
            f"Dataset downloaded: "
            f"{DATA_FILE.stat().st_size / (1024 * 1024):.1f} MB"
        )

    except Exception as exc:
        DATA_FILE.unlink(missing_ok=True)
        raise RuntimeError(
            "Dataset download failed. Check the Hugging Face file URL "
            "and the Streamlit Cloud logs."
        ) from exc

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
