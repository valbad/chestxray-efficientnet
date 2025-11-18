import os
import tarfile
import subprocess


def download_from_dropbox(dropbox_url, output_path):
    """
    Download a file from Dropbox using wget.
    """
    print(f"Downloading from: {dropbox_url}")
    subprocess.run(["wget", "-O", output_path, dropbox_url], check=True)


def extract_tar(tar_path, extract_dir):
    """
    Extract a .tar file to the given directory.
    """
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting {tar_path} to {extract_dir}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)

    print("Extraction completed.")


def download_and_prepare_dataset(suffix="?dl=1"):
    """
    Downloads and extracts the dataset from Dropbox.
    """
    # Paths inside Colab
    tar_path = "/content/chestxray_dataset.tar"
    extract_path = "/content/data"

    # Dropbox files
    train_tar_url = "https://www.dropbox.com/s/feaakvusdvkep3n/train.tar" + suffix
    csv_url = "https://www.dropbox.com/s/9kebfecemhfkj7k/train.csv" + suffix

    # Download train.tar
    download_from_dropbox(train_tar_url, tar_path)
    extract_tar(tar_path, extract_path)

    # Download train.csv (if needed)
    csv_output = os.path.join(extract_path, "train.csv")
    download_from_dropbox(csv_url, csv_output)

    print("\n✅ Dataset ready at:", extract_path)
    return extract_path


if __name__ == "__main__":
    download_and_prepare_dataset()
