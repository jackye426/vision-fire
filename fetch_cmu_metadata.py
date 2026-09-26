"""Download and verify CMU CREATE Lab's public smoke-video metadata snapshot."""

import hashlib
from pathlib import Path

import requests


TARGET = Path(__file__).resolve().parent / "dataset" / "cmu" / "metadata_02242020.json"
URL = ("https://raw.githubusercontent.com/CMU-CREATE-Lab/deep-smoke-machine/"
       "master/back-end/data/dataset/2020-02-24/metadata_02242020.json")
SHA256 = "cc85ad6db07557ae4afacc4f12f443b6e68ae0d88e30869fcf031f4c7dc7ee18"


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with TARGET.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
    with TARGET.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != SHA256:
        raise ValueError(f"Unexpected metadata SHA-256: {actual}")
    print(f"Verified {TARGET} ({TARGET.stat().st_size:,} bytes)")
