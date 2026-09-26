"""Run the released PyroNear temporal smoke package on six CMU clips.

Each CMU clip has 36 encoded frames representing about three minutes of
real-world time; adjacent frames are roughly five seconds apart. Clip labels
are smoke present/absent, without frame boxes or onset times.
"""

import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo-config"))

import cv2
from temporal_model.core.model import BboxTubeTemporalModel


MEDIA = ROOT / "dataset" / "cmu"
FRAMES = MEDIA / "temporal_frames"
PACKAGE = ROOT / "models" / "pyronear_temporal" / "model.zip"
OUT = ROOT / "outputs" / "cmu" / "pyronear_temporal_windows.csv"
COLUMNS = ("clip", "label", "camera_id", "window_start_frame", "window_end_frame",
           "detector_hit_frames", "kept_tubes", "max_tube_probability",
           "smoke_decision", "detector_ms_clip", "temporal_ms_window")


def extract(clip: str) -> list[Path]:
    folder = FRAMES / Path(clip).stem
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(MEDIA / "clips" / clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read {clip}")
    paths = []
    start = datetime(2026, 1, 1)
    frame = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        stamp = (start + timedelta(seconds=5 * frame)).strftime("%Y-%m-%dT%H-%M-%S")
        target = folder / f"{Path(clip).stem}_{stamp}.jpg"
        if not target.exists() and not cv2.imwrite(str(target), bgr):
            raise RuntimeError(f"Could not save {target}")
        paths.append(target)
        frame += 1
    cap.release()
    if len(paths) != 36:
        raise RuntimeError(f"Expected 36 frames, got {len(paths)} for {clip}")
    return paths


def main() -> None:
    with (MEDIA / "sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as handle:
            completed = {(r["clip"], int(r["window_start_frame"]))
                         for r in csv.DictReader(handle)}
    model = BboxTubeTemporalModel.from_package(PACKAGE, device="cpu")
    with OUT.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not completed:
            writer.writeheader()
        for item in manifest:
            clip = item["name"]
            spans = [(0, 20), (16, 36)]
            spans = [span for span in spans if (clip, span[0]) not in completed]
            if not spans:
                continue
            frames = model.load_sequence(extract(clip))
            started = time.perf_counter()
            detections = model.detect(frames)
            detector_ms = (time.perf_counter() - started) * 1000
            cache = {d.frame_id: d for d in detections}
            print(f"{clip}: companion detector {detector_ms:.0f}ms", flush=True)
            for first, end in spans:
                subset = frames[first:end]
                started = time.perf_counter()
                output = model.predict(subset, frame_detections=cache)
                temporal_ms = (time.perf_counter() - started) * 1000
                kept = output.details.get("tubes", {}).get("kept", [])
                probabilities = [float(t["probability"]) for t in kept
                                 if t.get("probability") is not None]
                row = {
                    "clip": clip, "label": item["label"],
                    "camera_id": item["camera_id"],
                    "window_start_frame": first, "window_end_frame": end - 1,
                    "detector_hit_frames": sum(bool(cache[f.frame_id].detections)
                                               for f in subset),
                    "kept_tubes": len(kept),
                    "max_tube_probability": max(probabilities, default=0),
                    "smoke_decision": int(output.is_positive),
                    "detector_ms_clip": round(detector_ms, 1),
                    "temporal_ms_window": round(temporal_ms, 1),
                }
                writer.writerow(row)
                handle.flush()
                print(f"  {first}-{end - 1}: decision={row['smoke_decision']} "
                      f"p={row['max_tube_probability']:.3f} tubes={len(kept)}", flush=True)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
