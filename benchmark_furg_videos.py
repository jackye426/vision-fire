"""Exploratory transfer check on five FURG videos with frame-level flame boxes.

FURG labels flame only. Smoke outputs are recorded but cannot be scored against
ground truth. Clips are sampled at approximately one source frame per second.
"""

import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
from PIL import Image

from detector import SPECS, detect


ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "dataset" / "furg"
OUT = ROOT / "outputs" / "furg" / "frame_results.csv"
CLIPS = ("barbecue", "hand_held_camera_wildfire", "house1",
         "non_fire_patrolbot_onboard", "coolerbot")
THRESHOLDS = (0.25, 0.50)
COLUMNS = ("clip", "model", "frame", "time_s", "flame_labeled", "gt_boxes",
           "latency_ms", "max_smoke", "max_flame", "predictions",
           "smoke_025", "flame_025", "flame_iou50_025",
           "smoke_050", "flame_050", "flame_iou50_050")


def flame_boxes(path: Path) -> dict[int, list[tuple[int, int, int, int]]]:
    result = {}
    for entry in ET.parse(path).getroot().find("frames"):
        number = int(entry.findtext("frameNumber"))
        boxes = []
        for item in entry.find("annotations"):
            values = list(map(int, item.text.split()))
            if len(values) == 4:
                x, y, w, h = values
                boxes.append((x, y, x + w, y + h))
        result[number] = boxes
    return result


def iou(a, b) -> float:
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    width = max(0, min(x2, u2) - max(x1, u1))
    height = max(0, min(y2, v2) - max(y1, v1))
    intersection = width * height
    area_a = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = max(0, u2 - u1) * max(0, v2 - v1)
    return intersection / max(area_a + area_b - intersection, 1)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as handle:
            completed = {(r["clip"], r["model"], int(r["frame"]))
                         for r in csv.DictReader(handle)}
    with OUT.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not completed:
            writer.writeheader()
        for clip in CLIPS:
            source = MEDIA / f"{clip}.mp4"
            gt = flame_boxes(MEDIA / f"{clip}.xml")
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot read {source}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = range(0, count, max(1, round(fps)))
            print(f"{clip}: {len(indices)} sampled frames, {fps:.2f} source fps", flush=True)
            for frame_index in indices:
                missing = [model for model in SPECS
                           if (clip, model, frame_index) not in completed]
                if not missing:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not read {clip} frame {frame_index}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                boxes = gt.get(frame_index, [])
                for model in missing:
                    started = time.perf_counter()
                    predictions = detect(model, image, floor=0.05)
                    latency = (time.perf_counter() - started) * 1000
                    maxima = {
                        label: max((d.score for d in predictions if d.label == label), default=0.0)
                        for label in ("smoke", "flame")
                    }
                    row = {
                        "clip": clip, "model": model, "frame": frame_index,
                        "time_s": round(frame_index / fps, 3),
                        "flame_labeled": int(bool(boxes)), "gt_boxes": len(boxes),
                        "latency_ms": round(latency, 1),
                        "max_smoke": round(maxima["smoke"], 6),
                        "max_flame": round(maxima["flame"], 6),
                        "predictions": json.dumps([{"label": d.label, "score": round(d.score, 4),
                                                    "box": d.box} for d in predictions]),
                    }
                    for threshold in THRESHOLDS:
                        suffix = f"{int(threshold * 100):03d}"
                        eligible_flames = [d for d in predictions
                                           if d.label == "flame" and d.score >= threshold]
                        row[f"smoke_{suffix}"] = int(maxima["smoke"] >= threshold)
                        row[f"flame_{suffix}"] = int(bool(eligible_flames))
                        row[f"flame_iou50_{suffix}"] = int(any(
                            iou(d.box, box) >= 0.50
                            for d in eligible_flames for box in boxes
                        ))
                    writer.writerow(row)
                    handle.flush()
            cap.release()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
