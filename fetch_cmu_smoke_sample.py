"""Fetch six public CMU industrial-smoke clips with concordant clip labels.

One strong-positive and one strong-negative clip are chosen per camera ID,
using the lowest metadata ID in each group. This is an exploratory sample,
not a held-out test or a prevalence-representative draw.
"""

import csv
import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "dataset" / "cmu"
METADATA = ROOT / "metadata_02242020.json"
CLIPS = ROOT / "clips"
MANIFEST = ROOT / "sample_manifest.csv"


def main() -> None:
    records = json.loads(METADATA.read_text(encoding="utf-8"))
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, code in (("smoke", 23), ("clear", 16)):
        for camera in (0, 1, 2):
            candidates = sorted((r for r in records
                                 if r["label_state"] == code
                                 and r["label_state_admin"] == code
                                 and r["camera_id"] == camera), key=lambda r: r["id"])
            if not candidates:
                raise RuntimeError(f"No concordant {label} clip for camera {camera}")
            record = candidates[0]
            url = (record["url_root"] + record["url_part"]).replace(
                "/180/", "/320/").replace("-180-180-", "-320-320-")
            name = f"{label}_cam{camera}_{record['id']}.mp4"
            target = CLIPS / name
            if not target.exists():
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if response.headers.get("Content-Type", "").split(";")[0] != "video/mp4":
                        raise RuntimeError(f"Unexpected media type for {url}")
                    with target.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
            with target.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            row = {"name": name, "label": label, "camera_id": camera,
                   "view_id": record["view_id"], "metadata_id": record["id"],
                   "label_state": code, "label_state_admin": code,
                   "source_url": url, "bytes": target.stat().st_size, "sha256": digest}
            rows.append(row)
            print(f"{name}: {row['bytes']:,} bytes", flush=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {MANIFEST}")


if __name__ == "__main__":
    main()
