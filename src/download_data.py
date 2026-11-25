import argparse
import shutil
import urllib.request
import zipfile

from config import DATA_RAW, FMA_METADATA_URL

WANTED = ["tracks.csv", "features.csv", "echonest.csv", "genres.csv"]

def download(dest_zip):
    print(f"Downloading {FMA_METADATA_URL} -> {dest_zip} (about 342 MB)...")

    def hook(blocks, block_size, total):
        if total > 0 and blocks % 500 == 0:
            print(f"  {min(blocks * block_size / total, 1):.0%}", end="\r")

    urllib.request.urlretrieve(FMA_METADATA_URL, dest_zip, reporthook=hook)
    print("\nDownload complete.")

def extract(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = member.split("/")[-1]
            if name in WANTED:
                with zf.open(member) as src, open(DATA_RAW / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  extracted {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        zip_path = DATA_RAW / "fma_metadata.zip"
        if not zip_path.exists():
            download(zip_path)
        extract(zip_path)

    missing = [f for f in WANTED if not (DATA_RAW / f).exists()]
    if missing:
        raise SystemExit(f"Missing files in {DATA_RAW}: {missing}")

    from data_loader import build_processed
    build_processed()


if __name__ == "__main__":
    main()