from pathlib import Path
import sys
import urllib.request
import zipfile
import argparse
from typing import Optional


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


def extract_zip_to_named_folder(zip_file_path: Path, password: Optional[str] = "objectnetisatestset") -> Path:
    target_dir = zip_file_path.with_suffix("")
    if target_dir.exists():
        print(f"Extraction target already exists: {target_dir}")
        return target_dir

    target_dir_resolved = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        members = zip_ref.infolist()

        # Safety check against Zip Slip and compute total size
        total_bytes = 0
        for member in members:
            resolved_member_path = (target_dir_resolved / member.filename).resolve()
            if not str(resolved_member_path).startswith(str(target_dir_resolved)):
                raise Exception(f"Unsafe path in zip entry: {member.filename}")
            if not member.is_dir():
                total_bytes += getattr(member, "file_size", 0)

        print(f"Extracting to: {target_dir}")
        pwd_bytes = password.encode("utf-8") if password else None

        bytes_written_total = 0
        chunk_size = 1024 * 1024  # 1 MiB

        for member in members:
            if member.is_dir():
                # Ensure directory exists
                (target_dir_resolved / member.filename).mkdir(parents=True, exist_ok=True)
                continue

            dest_path = (target_dir_resolved / member.filename)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member, pwd=pwd_bytes) as src, open(dest_path, "wb") as dst:
                while True:
                    data = src.read(chunk_size)
                    if not data:
                        break
                    dst.write(data)
                    bytes_written_total += len(data)

                    if total_bytes:
                        percent = bytes_written_total * 100.0 / total_bytes
                        progress = (
                            f"\rExtracting {target_dir.name}: "
                            f"{percent:.1f}% ({bytes_written_total/1_048_576:.1f} / {total_bytes/1_048_576:.1f} MiB)"
                        )
                    else:
                        progress = (
                            f"\rExtracting {target_dir.name}: "
                            f"{bytes_written_total/1_048_576:.1f} MiB"
                        )
                    print(progress, end="", flush=True)

        print(f"\nExtraction complete: {target_dir}")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract ObjectNet dataset zip")
    parser.add_argument(
        "--password",
        "-p",
        default="objectnetisatestset",
        help="Optional zip password. Default: objectnetisatestset",
    )
    args = parser.parse_args()

    try:
        zip_path = download_to_script_directory(OBJECTNET_URL)
        extract_zip_to_named_folder(zip_path, password=args.password)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()


