"""Compare local detectors on a fixed, balanced sample of D-Fire test images.

Run after fetch_dfire_splits.py and fetch_dfire_archive.py. Results are image
presence decisions, not object detection mAP. Scores are not calibrated across
models. The sample is deliberately balanced, not prevalence representative.
"""

import csv
import hashlib
import io
import random
import time
import zipfile
from pathlib import Path

from PIL import Image

from detector import SPECS, detect


ROOT = Path(__file__).parent
ARCHIVE = ROOT / "dataset" / "D-Fire.zip"
TEST_LIST = ROOT / "dataset" / "splits" / "dfire_test.txt"
OUT = ROOT / "outputs" / "dfire_benchmark"
SEED = 20260925
PER_CATEGORY = 40
CATEGORIES = ("none", "smoke_only", "flame_only", "both")
MODEL_NAMES = tuple(SPECS)
THRESHOLDS = (0.25, 0.5)


def category_for(label_data: bytes) -> str:
    classes = {int(line.split()[0]) for line in label_data.decode().splitlines() if line.strip()}
    if classes - {0, 1}:
        raise ValueError(f"Unexpected D-Fire class IDs: {classes}")
    return {(): "none", (0,): "smoke_only", (1,): "flame_only", (0, 1): "both"}[
        tuple(sorted(classes))
    ]


def sample_manifest(archive: zipfile.ZipFile) -> list[dict]:
    published_test = TEST_LIST.read_text(encoding="utf-8-sig").splitlines()
    if len(published_test) != 4306 or len(set(published_test)) != len(published_test):
        raise RuntimeError("Unexpected official D-Fire test list")
    entries = set(archive.namelist())
    groups = {category: [] for category in CATEGORIES}
    for name in published_test:
        image_path = f"test/images/{name}"
        label_path = f"test/labels/{Path(name).stem}.txt"
        if image_path not in entries or label_path not in entries:
            raise RuntimeError(f"Missing official test pair: {name}")
        groups[category_for(archive.read(label_path))].append(name)
    rng = random.Random(SEED)
    sample = []
    for category in CATEGORIES:
        for name in sorted(rng.sample(groups[category], PER_CATEGORY)):
            data = archive.read(f"test/images/{name}")
            sample.append({
                "name": name,
                "category": category,
                "source": "AoF" if name.startswith("AoF") else
                          "PublicDataset" if name.startswith("PublicDataset") else "WEB",
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    return sample


def write_manifest(rows: list[dict]) -> None:
    path = OUT / "image_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "category", "source", "sha256"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "image_results.csv"
    columns = ["name", "category", "source", "model", "latency_ms", "n_detections",
               "max_smoke", "max_flame", "smoke_025", "flame_025", "smoke_050", "flame_050"]
    completed = set()
    if result_path.exists():
        with result_path.open(newline="", encoding="utf-8") as handle:
            completed = {(row["name"], row["model"]) for row in csv.DictReader(handle)}
    with zipfile.ZipFile(ARCHIVE) as archive:
        manifest = sample_manifest(archive)
        write_manifest(manifest)
        with result_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if not completed:
                writer.writeheader()
            total = len(manifest) * len(MODEL_NAMES)
            done = len(completed)
            for item in manifest:
                data = archive.read(f"test/images/{item['name']}")
                image = Image.open(io.BytesIO(data)).convert("RGB")
                for model in MODEL_NAMES:
                    if (item["name"], model) in completed:
                        continue
                    started = time.perf_counter()
                    detections = detect(model, image, floor=0.05)
                    latency_ms = (time.perf_counter() - started) * 1000
                    maxima = {label: max((d.score for d in detections if d.label == label), default=0.0)
                              for label in ("smoke", "flame")}
                    row = {
                        "name": item["name"], "category": item["category"],
                        "source": item["source"], "model": model,
                        "latency_ms": f"{latency_ms:.1f}", "n_detections": len(detections),
                        "max_smoke": f"{maxima['smoke']:.6f}",
                        "max_flame": f"{maxima['flame']:.6f}",
                    }
                    for threshold in THRESHOLDS:
                        suffix = f"{int(threshold * 100):03d}"
                        for label in ("smoke", "flame"):
                            row[f"{label}_{suffix}"] = int(maxima[label] >= threshold)
                    writer.writerow(row)
                    handle.flush()
                    done += 1
                    if done % 25 == 0:
                        print(f"Image inferences: {done}/{total}", flush=True)
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
