from pathlib import Path
import requests
import zipfile
import io

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading NASA C-MAPSS dataset...")
    print(f"URL: {URL}")

    response = requests.get(URL, stream=True, timeout=120)
    response.raise_for_status()

    print("Download complete.")
    print("Extracting files...")

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall(RAW_DIR)

        print(f"Extracted {len(zip_file.namelist())} files.")

    print("\nDataset successfully loaded into:")
    print(RAW_DIR)


if __name__ == "__main__":
    download_and_extract()