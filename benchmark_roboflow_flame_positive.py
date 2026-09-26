"""Run four flame-capable detectors on distinct Fire Data Annotations test images.

Reports positive-image hit rate and image-level box alignment, not precision,
specificity, mAP, or an incident-level generalization estimate. Resumable.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

from PIL import Image

from detector import SPECS, detect


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "dataset" / "roboflow" / "fire-data-annotations-v5-yolov8.zip"
OUT = ROOT / "outputs" / "roboflow_flame_positive"
MODELS = ("dfire_yolov8n", "dfine", "yolo", "soul")
THRESHOLDS = (0.25, 0.50)
FIELDS = (
    "image", "model", "sha256", "width", "height", "gt_boxes", "largest_gt_fraction",
    "latency_ms", "flame_candidates", "max_flame_score",
    "hit_025", "aligned30_025", "aligned50_025", "best_iou_025",
    "hit_050", "aligned30_050", "aligned50_050", "best_iou_050",
)


def source_id(path: str) -> str:
    stem = Path(path).stem.split(".rf.", 1)[0]
    stem = re.sub(r"_jpg$", "", stem, flags=re.I)
    return re.sub(r"^(?:mirror|noise)+", "", stem, flags=re.I).lower()


def clean_test_images(entries: list[str]) -> tuple[list[str], dict]:
    train_sources = {source_id(n) for n in entries if n.startswith("train/images/")
                     and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    test = sorted(n for n in entries if n.startswith("test/images/")
                  and n.lower().endswith((".jpg", ".jpeg", ".png")))
    candidates = [n for n in test if source_id(n) not in train_sources]
    by_source = {}
    for name in candidates:
        sid = source_id(name)
        # Prefer an original-looking file over an explicit Mirror/Noise variant.
        original = not re.match(r"^(?:mirror|noise)", Path(name).name, flags=re.I)
        if sid not in by_source or (original and not by_source[sid][0]):
            by_source[sid] = (original, name)
    selected = sorted(value[1] for value in by_source.values())
    metadata = {
        "published_test_images": len(test),
        "test_images_sharing_source_id_with_train": len(test) - len(candidates),
        "test_images_after_train_overlap_filter": len(candidates),
        "one_image_per_remaining_source_id": len(selected),
        "source_id_rule": "lowercase filename, strip _jpg and leading Mirror/Noise prefixes",
    }
    return selected, metadata


def iou(a: tuple[int, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def label_boxes(text: str, width: int, height: int) -> list[tuple[float, ...]]:
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Unexpected label: {line}")
        cx, cy, bw, bh = map(float, parts[1:])
        if not all(0 <= n <= 1 for n in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"Invalid coordinates: {line}")
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height))
    if not boxes:
        raise ValueError("Expected at least one flame box in every test image")
    return boxes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = OUT / "per_image.csv"
    completed = set()
    if results.exists():
        with results.open(newline="", encoding="utf-8") as handle:
            completed = {(r["image"], r["model"]) for r in csv.DictReader(handle)}
    with zipfile.ZipFile(ARCHIVE) as archive:
        images, cohort = clean_test_images(archive.namelist())
        if cohort["published_test_images"] != 663 or not images:
            raise RuntimeError(f"Unexpected test set: {cohort}")
        (OUT / "cohort.json").write_text(json.dumps(cohort, indent=2), encoding="utf-8")
        (OUT / "cohort_images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
        entries = set(archive.namelist())
        with results.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not results.stat().st_size:
                writer.writeheader()
            selected = set(images)
            done = sum(image in selected for image, _ in completed)
            total = len(images) * len(MODELS)
            for path in images:
                missing = [model for model in MODELS if (path, model) not in completed]
                if not missing:
                    continue
                label_path = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                if label_path not in entries:
                    raise RuntimeError(f"Missing label: {label_path}")
                data = archive.read(path)
                image = Image.open(io.BytesIO(data)).convert("RGB")
                width, height = image.size
                boxes = label_boxes(archive.read(label_path).decode(), width, height)
                largest_gt = max((b[2] - b[0]) * (b[3] - b[1]) / (width * height) for b in boxes)
                sha256 = hashlib.sha256(data).hexdigest()
                for model in missing:
                    if "flame" not in SPECS[model]["labels"]:
                        raise RuntimeError(f"Model has no flame class: {model}")
                    started = time.perf_counter()
                    detections = [d for d in detect(model, image, floor=0.05) if d.label == "flame"]
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    row = {
                        "image": path, "model": model, "sha256": sha256,
                        "width": width, "height": height, "gt_boxes": len(boxes),
                        "largest_gt_fraction": f"{largest_gt:.7f}",
                        "latency_ms": f"{elapsed_ms:.1f}",
                        "flame_candidates": len(detections),
                        "max_flame_score": f"{max((d.score for d in detections), default=0):.6f}",
                    }
                    for threshold in THRESHOLDS:
                        suffix = f"{int(threshold * 100):03d}"
                        candidates = [d for d in detections if d.score >= threshold]
                        best = max((iou(d.box, b) for d in candidates for b in boxes), default=0.0)
                        row[f"hit_{suffix}"] = int(bool(candidates))
                        row[f"aligned30_{suffix}"] = int(best >= 0.30)
                        row[f"aligned50_{suffix}"] = int(best >= 0.50)
                        row[f"best_iou_{suffix}"] = f"{best:.6f}"
                    writer.writerow(row)
                    handle.flush()
                    done += 1
                    if done % 100 == 0:
                        print(f"Completed {done}/{total} model-image pairs", flush=True)
    print(f"Saved {results}", flush=True)


if __name__ == "__main__":
    main()
