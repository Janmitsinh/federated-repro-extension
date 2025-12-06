"""Require to download dataset or additional preparation."""

from pathlib import Path
import shutil
import urllib.request
import tarfile

def download_imagenette(url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"):
    """Download and extract Imagenette dataset."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_dir / "imagenette2-160"
    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}")
        return

    tar_path = data_dir / "imagenette2-160.tgz"
    
    print(f"Downloading Imagenette from {url}...")
    urllib.request.urlretrieve(url, tar_path)
    
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
        
    # Remove tar file
    tar_path.unlink()
    print("Done!")

if __name__ == "__main__":
    download_imagenette()
