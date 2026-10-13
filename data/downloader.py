import os
import requests


def download_official_zenodo_file(
    record_id: str, filename: str, output_dir: str = "data/sample"
) -> str:
    """Downloads official research datasets directly from CERN's Zenodo repository REST API."""
    os.makedirs(output_dir, exist_ok=True)
    destination_path = os.path.join(output_dir, filename)

    # Official Zenodo direct file URL format
    url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"

    print(
        f"🌐 Connecting directly to official Zenodo CERN repository (Record: {record_id})..."
    )
    print(f"⬇️ Downloading: {filename}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    done = int(50 * downloaded / total_size)
                    print(
                        f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB",
                        end="",
                    )

    print(f"\n✅ Successfully downloaded from official source: {destination_path}")
    return destination_path


if __name__ == "__main__":
    print("Downloader module initialized.")