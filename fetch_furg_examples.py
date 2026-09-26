"""Fetch two public FURG fire videos and their flame-box annotations.

FURG repository: https://github.com/steffensbola/furg-fire-dataset (CC0-1.0).
The files are independent demonstration media, not part of the D-Fire dataset.
"""

import hashlib
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "dataset" / "furg"
BASE = "https://raw.githubusercontent.com/steffensbola/furg-fire-dataset/master/"
FILES = ("house1.mp4", "house1.xml", "hand_held_camera_wildfire.mp4",
         "hand_held_camera_wildfire.xml", "barbecue.mp4", "barbecue.xml",
         "non_fire_patrolbot_onboard.mp4", "non_fire_patrolbot_onboard.xml",
         "coolerbot.mp4", "coolerbot.xml")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = ROOT / name
        if not target.exists():
            with requests.get(BASE + name, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        print(f"{name}: {target.stat().st_size:,} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
