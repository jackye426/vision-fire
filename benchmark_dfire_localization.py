"""Check whether positive image detections overlap D-Fire's labeled boxes.

For each class present in a sampled image, count an image as localized when at
least one predicted box of that class overlaps one ground-truth box at IoU >=
0.30 or 0.50. This is image-level localization recall, not COCO mAP.
"""

import csv
import io
import zipfile
from pathlib import Path

from PIL import Image

from detector import SPECS, detect


ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "dfire_benchmark"
THRESHOLD = 0.25
CLASS_NAMES = {0: "smoke", 1: "flame"}


def iou(a, b) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def ground_truth(text: str, width: int, height: int) -> dict[str, list[tuple[float, ...]]]:
    result = {"smoke": [], "flame": []}
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split()
        class_id = int(values[0])
        cx, cy, bw, bh = map(float, values[1:5])
        result[CLASS_NAMES[class_id]].append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                                               (cx + bw / 2) * width, (cy + bh / 2) * height))
    return result


def main() -> None:
    with (OUT / "image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    with (OUT / "image_results.csv").open(newline="", encoding="utf-8") as handle:
        scores = {(r["name"], r["model"]): r for r in csv.DictReader(handle)}
    output_path = OUT / "localization_results.csv"
    columns = ("name", "category", "model", "class", "raw_presence", "best_iou",
               "localized_iou30", "localized_iou50")
    completed = set()
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            completed = {(r["name"], r["model"], r["class"]) for r in csv.DictReader(handle)}
    with zipfile.ZipFile(ROOT / "dataset" / "D-Fire.zip") as archive, \
            output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not completed:
            writer.writeheader()
        processed = 0
        for item in manifest:
            if item["category"] == "none":
                continue
            name = item["name"]
            image = Image.open(io.BytesIO(archive.read(f"test/images/{name}"))).convert("RGB")
            width, height = image.size
            labels = ground_truth(archive.read(f"test/labels/{Path(name).stem}.txt").decode(),
                                  width, height)
            for model in SPECS:
                applicable = [label for label in CLASS_NAMES.values()
                              if labels[label] and label in SPECS[model]["labels"]
                              and (name, model, label) not in completed]
                if not applicable:
                    continue
                if any(float(scores[(name, model)][f"max_{label}"]) >= THRESHOLD
                       for label in applicable):
                    found = detect(model, image, floor=THRESHOLD)
                else:
                    found = []
                for label in applicable:
                    candidates = [d for d in found if d.label == label and d.score >= THRESHOLD]
                    best = max((iou(d.box, target) for d in candidates for target in labels[label]),
                               default=0.0)
                    writer.writerow({"name": name, "category": item["category"],
                                     "model": model, "class": label,
                                     "raw_presence": int(bool(candidates)),
                                     "best_iou": f"{best:.4f}",
                                     "localized_iou30": int(best >= 0.30),
                                     "localized_iou50": int(best >= 0.50)})
                    handle.flush()
                    processed += 1
                    if processed % 50 == 0:
                        print(f"Localization rows: {processed}", flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
