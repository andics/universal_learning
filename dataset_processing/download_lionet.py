import requests
import os
from tqdm import tqdm

def download_zip(
    url: str = "http://manitoulin.csail.mit.edu:8023/laionet_folders.zip",
    save_dir: str = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/",
    filename: str = "laionet_folders.zip"
):
    """
    Downloads a ZIP file from a URL into a specific directory with a progress bar
    and verbose text output every ~3% of completion.
    """

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    # Stream download
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total size in bytes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 KB chunks

    # Progress bar setup
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")

    # For ~3% logging
    percent_step = 3
    next_percent = percent_step

    downloaded = 0
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                progress_bar.update(len(chunk))

                # Calculate percentage
                percent = (downloaded / total_size) * 100
                if percent >= next_percent:
                    print(f"Download progress: {percent:.2f}%")
                    next_percent += percent_step

    progress_bar.close()
    print(f"\n✅ Download complete! File saved at: {file_path}")
    return file_path


# Example usage with your defaults
if __name__ == "__main__":
    download_zip()
