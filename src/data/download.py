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

    # we start by determining if we need to use the hackathon's dropbox urls to download the train/test data.
    subfolders = [f.lower() for f in os.listdir(data_dir) if len(f.split(".")) == 1]
    need_download_train, need_download_test = True, True
    for f in subfolders: 
      if "train" in f:
        need_download_train = False
      if "test" in f:
        need_download_test = False

    if need_download_train: 
      # downloading train data
      print(f"Downaloading train data directory...")
      tar_path = os.path.join(data_dir, "train.tar")
      download_from_dropbox(train_tar_url + suffix, tar_path)
      extract_tar(tar_path,data_dir)

      # downloading train csv
      download_from_dropbox(csv_url + suffix, csv_path)
      print("✅ Training data and train.csv downloaded.")

    if need_download_test:
      if test_tar_url:
        # downloading test data
        print(f"Downaloading test data directory...")
        tar_path = os.path.join(data_dir, "test.tar")
        download_from_dropbox(test_tar_url + suffix, tar_path)
        extract_tar(tar_path, data_dir)
        print("✅ Test data downloaded.")
      else:
        print("No test tar URL provided!")

    if not need_download_test and not need_download_train:
      print("✔ data/train already populated and train.csv found.")
      print("✔ data/test already populated.")

    if not os.path.isfile(csv_path): #worst case scenario: train data is there but not train csv...
          raise FileNotFoundError(
          f"data/train is not empty but train.csv is missing at {csv_path}. "
          "Provide a CSV or empty the folder to trigger automatic download."
        )
            
