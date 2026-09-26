"""Run PyroNear's released YOLO + tube classifier on D-Fire test clips.

Requires the isolated Python 3.12 `.venv-temporal` with temporal-model-core
installed. See TEMPORAL_EXPERIMENT.md. Windows are up to 20 frames at ~1 fps;
overlapping windows reuse the companion detector's per-frame predictions.
"""

import argparse
import csv
import os
from pathlib import Path
import time
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo-config"))

import cv2
from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.stage_timer import StageTimer


PACKAGE = ROOT / "models" / "pyronear_temporal" / "model.zip"
VIDEO_DIR = ROOT / "dataset" / "videos"
FRAME_DIR = ROOT / "dataset" / "temporal_frames"
OUT = ROOT / "outputs" / "dfire_benchmark" / "pyronear_temporal_windows.csv"
SAMPLE = ("FP1.mp4", "FP14.mp4", "FP31.mp4", "VP1.mp4", "VP3.mp4", "VP5.mp4")
COLUMNS = ("video", "window_start_s", "window_end_s", "frames", "detector_hits",
           "candidate_tubes", "kept_tubes", "smoke_decision", "max_tube_probability",
           "detector_ms_clip", "tube_pipeline_ms", "pad_ms", "tubes_ms", "crop_ms",
           "classifier_ms", "trigger_ms")


def extract_frames(name: str) -> list[Path]:
    target_dir = FRAME_DIR / Path(name).stem
    target_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_DIR / name))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        raise RuntimeError(f"Bad video metadata for {name}")
    start = datetime(2026, 1, 1)
    paths = []
    for second, frame_index in enumerate(range(0, total, max(1, round(fps)))):
        stamp = (start + timedelta(seconds=second)).strftime("%Y-%m-%dT%H-%M-%S")
        target = target_dir / f"{Path(name).stem}_{stamp}.jpg"
        if not target.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = cap.read()
            if not ok or not cv2.imwrite(str(target), bgr):
                raise RuntimeError(f"Could not extract {name} frame {frame_index}")
        paths.append(target)
    cap.release()
    return paths


def windows(length: int) -> list[tuple[int, int]]:
    if length <= 20:
        return [(0, length)]
    return [(start, min(start + 20, length)) for start in range(0, length - 19, 10)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", choices=SAMPLE, help="Run one test clip; omit for all six")
    args = parser.parse_args()
    names = (args.video,) if args.video else SAMPLE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as handle:
            existing = {(r["video"], int(r["window_start_s"]))
                        for r in csv.DictReader(handle)}
    started = time.perf_counter()
    model = BboxTubeTemporalModel.from_package(PACKAGE, device="cpu")
    print(f"Loaded PyroNear v0.4.0 on CPU in {time.perf_counter() - started:.1f}s", flush=True)
    with OUT.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not existing:
            writer.writeheader()
        for name in names:
            paths = extract_frames(name)
            spans = [span for span in windows(len(paths)) if (name, span[0]) not in existing]
            if not spans:
                continue
            frames = model.load_sequence(paths)
            started = time.perf_counter()
            detections = model.detect(frames)
            detector_ms = (time.perf_counter() - started) * 1000
            cache = {d.frame_id: d for d in detections}
            print(f"{name}: {len(frames)} frames, companion detector {detector_ms:.0f}ms",
                  flush=True)
            for first, end in spans:
                subset = frames[first:end]
                timer = StageTimer()
                started = time.perf_counter()
                result = model.predict(subset, frame_detections=cache, timer=timer)
                pipeline_ms = (time.perf_counter() - started) * 1000
                details = result.details
                kept = details.get("tubes", {}).get("kept", [])
                probabilities = [float(t["probability"]) for t in kept
                                 if t.get("probability") is not None]
                timings = timer.as_dict()
                row = {
                    "video": name, "window_start_s": first, "window_end_s": end - 1,
                    "frames": len(subset),
                    "detector_hits": sum(bool(cache[f.frame_id].detections) for f in subset),
                    "candidate_tubes": details.get("tubes", {}).get("num_candidates", 0),
                    "kept_tubes": len(kept), "smoke_decision": int(result.is_positive),
                    "max_tube_probability": max(probabilities, default=0.0),
                    "detector_ms_clip": round(detector_ms, 1),
                    "tube_pipeline_ms": round(pipeline_ms, 1),
                    "pad_ms": round(timings.get("pad", 0), 1),
                    "tubes_ms": round(timings.get("tubes", 0), 1),
                    "crop_ms": round(timings.get("crop", 0), 1),
                    "classifier_ms": round(timings.get("classifier", 0), 1),
                    "trigger_ms": round(timings.get("trigger_search", 0), 1),
                }
                writer.writerow(row)
                handle.flush()
                print(f"  {first}-{end - 1}s: smoke={result.is_positive}, "
                      f"p={row['max_tube_probability']:.3f}, tubes={len(kept)}, "
                      f"temporal={pipeline_ms:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
