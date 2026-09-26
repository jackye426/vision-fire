"""Run the five local frame detectors on user-selected YouTube clips at 1 fps.

The clips have user-supplied approximate onset notes, not frame-level labels.
This script preserves detections for later visual review and event-level analysis.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
from PIL import Image

from detector import SPECS, detect


ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "dataset" / "youtube"
OUT = ROOT / "outputs" / "youtube" / "frame_results.csv"
CLIPS = ("CI6YpclCYA4", "WsUjSE-ibKo", "ZBrOzQRoI-E", "5NAkyEmC0IU",
         "K3ML54RtbAo", "wm22sUiq9fE")
MODELS = ("dfire_yolov8n", "dfine", "yolo", "soul", "pyronear")
FIELDS = ("clip", "model", "time_s", "frame_index", "width", "height", "latency_ms",
          "max_smoke", "max_flame", "smoke_025", "flame_025", "smoke_050", "flame_050",
          "detections")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as handle:
            completed = {(r["clip"], r["model"], int(r["time_s"])) for r in csv.DictReader(handle)}
    with OUT.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not OUT.stat().st_size:
            writer.writeheader()
        for clip in CLIPS:
            source = MEDIA / f"{clip}.mp4"
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {source}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            seconds = range(int((frames - 1) / fps) + 1)
            print(f"{clip}: {len(seconds)} sampled seconds; {fps:.3f} fps", flush=True)
            for second in seconds:
                missing = [m for m in MODELS if (clip, m, second) not in completed]
                if not missing:
                    continue
                frame_index = min(round(second * fps), frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {clip} at {second}s/frame {frame_index}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                width, height = image.size
                for model in missing:
                    started = time.perf_counter()
                    found = detect(model, image, floor=0.05)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    maxima = {label: max((d.score for d in found if d.label == label), default=0.0)
                              for label in ("smoke", "flame")}
                    row = {
                        "clip": clip, "model": model, "time_s": second,
                        "frame_index": frame_index, "width": width, "height": height,
                        "latency_ms": f"{elapsed_ms:.1f}",
                        "max_smoke": f"{maxima['smoke']:.6f}",
                        "max_flame": f"{maxima['flame']:.6f}",
                        "smoke_025": int(maxima["smoke"] >= 0.25),
                        "flame_025": int(maxima["flame"] >= 0.25),
                        "smoke_050": int(maxima["smoke"] >= 0.50),
                        "flame_050": int(maxima["flame"] >= 0.50),
                        "detections": json.dumps([{"label": d.label, "score": round(d.score, 5),
                                                   "box": d.box} for d in found]),
                    }
                    writer.writerow(row)
                    handle.flush()
                if second % 10 == 0:
                    print(f"{clip}: through {second}s", flush=True)
            cap.release()
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
