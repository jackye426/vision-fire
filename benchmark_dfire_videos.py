"""Run every local detector on one frame per second of six D-Fire test videos.

The official video split supplies clip names, not frame-level fire/smoke labels or
onset timestamps. This script records detections and latency, not video accuracy.
"""

import csv
import time
from pathlib import Path

import cv2
from PIL import Image

from detector import SPECS, detect
from fetch_dfire_videos import SAMPLE


ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "dfire_benchmark"
THRESHOLDS = (0.25, 0.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "video_frames.csv"
    columns = ["video", "model", "frame", "time_s", "width", "height",
               "latency_ms", "max_smoke", "max_flame", "smoke_025", "flame_025",
               "smoke_050", "flame_050"]
    completed = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            completed = {(r["video"], r["model"], int(r["frame"]))
                         for r in csv.DictReader(handle)}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not completed:
            writer.writeheader()
        for name in SAMPLE:
            video_path = ROOT / "dataset" / "videos" / name
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count <= 0:
                raise RuntimeError(f"Invalid video metadata for {name}")
            step = max(1, round(fps))
            sampled = range(0, count, step)
            print(f"{name}: {len(sampled)} frames at ~1 fps", flush=True)
            for frame_index in sampled:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, bgr = cap.read()
                if not success:
                    raise RuntimeError(f"Could not read {name} frame {frame_index}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                height, width = bgr.shape[:2]
                for model in SPECS:
                    if (name, model, frame_index) in completed:
                        continue
                    started = time.perf_counter()
                    detections = detect(model, image, floor=0.05)
                    latency_ms = (time.perf_counter() - started) * 1000
                    maxima = {label: max((d.score for d in detections if d.label == label), default=0.0)
                              for label in ("smoke", "flame")}
                    row = {"video": name, "model": model, "frame": frame_index,
                           "time_s": f"{frame_index / fps:.3f}",
                           "width": width, "height": height,
                           "latency_ms": f"{latency_ms:.1f}",
                           "max_smoke": f"{maxima['smoke']:.6f}",
                           "max_flame": f"{maxima['flame']:.6f}"}
                    for threshold in THRESHOLDS:
                        suffix = f"{int(threshold * 100):03d}"
                        for label in ("smoke", "flame"):
                            row[f"{label}_{suffix}"] = int(maxima[label] >= threshold)
                    writer.writerow(row)
                    handle.flush()
            cap.release()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
