# src/data/download.py

import os
import tarfile
import subprocess


def download_from_dropbox(dropbox_url, output_path):
    """
    Download a file from Dropbox using wget.
    """
    print(f"⬇️  Downloading from: {dropbox_url}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.run(["wget", "-O", output_path, dropbox_url], check=True)
    print(f"   Saved to: {output_path}")


def extract_tar(tar_path, extract_dir):
    """
    Extract a .tar file to the given directory.
    """
    os.makedirs(extract_dir, exist_ok=True)
    print(f"📦 Extracting {tar_path} → {extract_dir} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    print("   Extraction completed.")


def ensure_data_dirs(cfg_data):
    """
    1) Create data/train and data/test folders if missing.
    2) If data is empty, download from Dropbox (train.tar, test.tar, train.csv).
    3) If train.csv is missing → raise error.
    """
    data_dir   = cfg_data["data_dir"]      # ex: "data"
    csv_path   = cfg_data["train_csv"]     # ex: "data/train.csv"

    train_tar_url = cfg_data["dropbox_train_tar"]   # ex: .../train.tar
    test_tar_url  = cfg_data["dropbox_test_tar"]    # ex: .../testPublic.tar
    csv_url       = cfg_data["dropbox_train_csv"]   # ex: .../train.csv
    suffix        = cfg_data.get("suffix", "?dl=1")

    os.makedirs(data_dir, exist_ok=True)

    def folder_empty(path):
        return len(os.listdir(path)) == 0

    # --- TRAIN folder ---
    if folder_empty(data_dir):
        print(f"⚠ data directory is empty → downloading all dataset...")
        tar_path = os.path.join(data_dir, "train.tar")
        download_from_dropbox(train_tar_url + suffix, tar_path)
        extract_tar(tar_path, data_dir)

        # train.csv
        download_from_dropbox(csv_url + suffix, csv_path)
        print("✅ Training data and train.csv downloaded.")
        if test_tar_url:
          tar_path = os.path.join(data_dir, "test.tar")
          download_from_dropbox(test_tar_url + suffix, tar_path)
          extract_tar(tar_path, data_dir)
          print("✅ Test data downloaded.")
        else:
          print("ℹ No test tar URL provided.")
    else:
        # Train not empty → we require train.csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"data/train is not empty but train.csv is missing at {csv_path}. "
                "Provide a CSV or empty the folder to trigger automatic download."
            )
        else:
            folders = [f for f in os.listdir(data_dir) if len(f.split(".")) == 1]
            exist_train = False
            exist_test = False
            for f in folders: 
              if "train" in f:
                exist_train = True
              if "test" in f: 
                exist_test = True
              if exist_train and exist_test: 
                print("✔ data/train already populated and train.csv found.")
                print("✔ data/test already populated.")
                break     
            if not exist_train and not exist_test: 
              print("Something went wrong, please refer to README.md and check datastructure")
            
