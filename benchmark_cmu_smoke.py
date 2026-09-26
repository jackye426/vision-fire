"""Exploratory frame-detector comparison on six CMU industrial-smoke clips.

CMU provides clip-level smoke labels, not frame boxes or onset times. All 36
frames per 320px clip are evaluated; a clip hit means any smoke box appears.
"""

import csv
import json
from pathlib import Path
import statistics
import time

import cv2
from PIL import Image

from detector import SPECS, detect


ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "dataset" / "cmu"
OUT = ROOT / "outputs" / "cmu"
FRAME_CSV = OUT / "frame_results.csv"
SUMMARY_CSV = OUT / "clip_summary.csv"
THRESHOLDS = (0.25, 0.50)
FRAME_COLUMNS = ("clip", "label", "camera_id", "model", "frame", "time_s",
                 "latency_ms", "max_smoke", "max_flame", "predictions")


def first_two(flags: list[bool]) -> int | None:
    return next((i for i in range(1, len(flags)) if flags[i - 1] and flags[i]), None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (MEDIA / "sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    completed = set()
    if FRAME_CSV.exists():
        with FRAME_CSV.open(newline="", encoding="utf-8") as handle:
            completed = {(r["clip"], r["model"], int(r["frame"]))
                         for r in csv.DictReader(handle)}
    with FRAME_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS)
        if not completed:
            writer.writeheader()
        for item in manifest:
            source = MEDIA / "clips" / item["name"]
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot read {source}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"{item['name']}: {count} frames", flush=True)
            for frame_index in range(count):
                missing = [model for model in SPECS
                           if (item["name"], model, frame_index) not in completed]
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read {source} frame {frame_index}")
                if not missing:
                    continue
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                for model in missing:
                    started = time.perf_counter()
                    detections = detect(model, image, floor=0.05)
                    latency = (time.perf_counter() - started) * 1000
                    scores = {label: max((d.score for d in detections if d.label == label),
                                         default=0.0) for label in ("smoke", "flame")}
                    writer.writerow({
                        "clip": item["name"], "label": item["label"],
                        "camera_id": item["camera_id"], "model": model,
                        "frame": frame_index, "time_s": round(frame_index / fps, 3),
                        "latency_ms": round(latency, 1),
                        "max_smoke": round(scores["smoke"], 6),
                        "max_flame": round(scores["flame"], 6),
                        "predictions": json.dumps([{"label": d.label,
                                                    "score": round(d.score, 4),
                                                    "box": d.box} for d in detections]),
                    })
                    handle.flush()
            cap.release()
    with FRAME_CSV.open(newline="", encoding="utf-8") as handle:
        frames = list(csv.DictReader(handle))
    rows = []
    for item in manifest:
        for model in SPECS:
            group = sorted((r for r in frames if r["clip"] == item["name"]
                            and r["model"] == model), key=lambda r: int(r["frame"]))
            if len(group) != 36:
                raise RuntimeError(f"Incomplete {item['name']} / {model}: {len(group)} frames")
            for threshold in THRESHOLDS:
                smoke = [float(r["max_smoke"]) >= threshold for r in group]
                flame = [float(r["max_flame"]) >= threshold for r in group]
                rows.append({
                    "clip": item["name"], "label": item["label"],
                    "camera_id": item["camera_id"], "model": model,
                    "threshold": threshold, "frames": len(group),
                    "smoke_flagged_frames": sum(smoke),
                    "flame_flagged_frames": sum(flame),
                    "smoke_any": int(any(smoke)),
                    "smoke_two_consecutive": int(first_two(smoke) is not None),
                    "flame_any": int(any(flame)),
                    "median_cpu_ms": round(statistics.median(float(r["latency_ms"])
                                                            for r in group), 1),
                })
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {FRAME_CSV} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
