import os
import requests
from tqdm import tqdm


def download_file(url: str, file_name: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))

    with open(file_name, "wb") as file, tqdm(
        desc=file_name,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_dataset():
    output_file = os.path.join(os.getcwd(), "train.txt")
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

    if os.path.exists(output_file):
        print(f"Dataset already exists at {output_file}")
        return output_file

    print(f"Downloading TinyStories dataset to {output_file}...")
    download_file(url, output_file)
    print(f"Saved dataset to {output_file}")

    return output_file


if __name__ == "__main__":
    download_dataset()
