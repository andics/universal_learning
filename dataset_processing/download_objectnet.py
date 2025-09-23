import os
from pathlib import Path
import sys
import urllib.request


OBJECTNET_URL = "https://objectnet.dev/downloads/objectnet-1.0.zip"


def download_to_script_directory(url: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    filename = url.rsplit("/", 1)[-1]
    destination_path = script_dir / filename

    if destination_path.exists():
        print(f"File already exists: {destination_path}")
        return destination_path

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request) as response:
            total_size_header = response.getheader("Content-Length")
            total_size = int(total_size_header) if total_size_header else None

            bytes_downloaded = 0
            chunk_size = 1024 * 1024  # 1 MiB chunks
            destination_path_tmp = destination_path.with_suffix(destination_path.suffix + ".part")

            with open(destination_path_tmp, "wb") as file_handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    file_handle.write(chunk)
                    bytes_downloaded += len(chunk)

                    if total_size:
                        percent = bytes_downloaded * 100.0 / total_size
                        progress = (
                            f"\rDownloading {destination_path.name}: "
                            f"{percent:.1f}% ({bytes_downloaded/1_048_576:.1f} / {total_size/1_048_576:.1f} MiB)"
                        )
                    else:
                        progress = (
                            f"\rDownloading {destination_path.name}: "
                            f"{bytes_downloaded/1_048_576:.1f} MiB"
                        )
                    print(progress, end="", flush=True)

            # Move temp file to final destination when complete
            destination_path_tmp.replace(destination_path)
            print(f"\nDownloaded to: {destination_path}")
            return destination_path

    except Exception as error:  # noqa: BLE001 - print and cleanup for any download error
        # Clean up partial download if exists
        try:
            destination_path_tmp = destination_path.with_suffix(destination_path.suffix + ".part")
            if destination_path_tmp.exists():
                destination_path_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"Error downloading file: {error}")
        raise


def main() -> None:
    try:
        download_to_script_directory(OBJECTNET_URL)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()


