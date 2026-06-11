"""Download the HAM10000 dataset from Kaggle.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(https://www.kaggle.com/docs/api#authentication).

Usage:
    python scripts/download_data.py --data-dir data/ham10000
"""
import argparse
from pathlib import Path

DATASET = "kmader/skin-cancer-mnist-ham10000"
METADATA = "HAM10000_metadata.csv"


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / METADATA).exists():
        print(f"Dataset already present in {data_dir}, skipping download.")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {DATASET} -> {data_dir}")
    api.dataset_download_files(DATASET, path=str(data_dir), unzip=True)
    print(f"Done. Metadata at {data_dir / METADATA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ham10000"),
                        help="Target directory for the dataset (default: data/ham10000)")
    download(parser.parse_args().data_dir)
