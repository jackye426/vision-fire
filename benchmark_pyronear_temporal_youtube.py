"""Run PyroNear's released smoke temporal pipeline on user-selected clips.

One frame per playback second; 20-frame windows with 10-frame stride, plus a
tail window. The two waste-pit camera angles are kept in separate segments.
The edited compilation is excluded because its scene cuts break tube meaning.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo-config"))

import cv2
from temporal_model.core.model import BboxTubeTemporalModel


PACKAGE = ROOT / "models" / "pyronear_temporal" / "model.zip"
MEDIA = ROOT / "dataset" / "youtube"
FRAMES = MEDIA / "temporal_frames"
OUT = ROOT / "outputs" / "youtube" / "pyronear_temporal_windows.csv"
SEGMENTS = (
    ("CI6YpclCYA4", "battery_fire", 0, 19),
    ("WsUjSE-ibKo", "recycling", 0, 93),
    ("5NAkyEmC0IU", "angle1", 0, 52),
    ("5NAkyEmC0IU", "angle2", 53, 102),
    ("K3ML54RtbAo", "screen_recording", 0, 38),
    ("wm22sUiq9fE", "screen_recording", 0, 26),
)
FIELDS = ("clip", "segment", "window_start_s", "window_end_s", "frames",
          "detector_hit_frames", "candidate_tubes", "kept_tubes",
          "max_tube_probability", "smoke_decision", "detector_ms_clip",
          "temporal_ms_window")


def extract(clip: str) -> list[Path]:
    folder = FRAMES / clip
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(MEDIA / f"{clip}.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = range(int((count - 1) / fps) + 1)
    paths = []
    anchor = datetime(2026, 1, 1)
    try:
        for second in seconds:
            stamp = (anchor + timedelta(seconds=second)).strftime("%Y-%m-%dT%H-%M-%S")
            target = folder / f"{clip}_{stamp}.jpg"
            if not target.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(round(second * fps), count - 1))
                ok, bgr = cap.read()
                if not ok or not cv2.imwrite(str(target), bgr):
                    raise RuntimeError(f"Cannot extract {clip} at {second}s")
            paths.append(target)
    finally:
        cap.release()
    return paths


def windows(start: int, end: int) -> list[tuple[int, int]]:
    """Return half-open 20-frame spans covering a segment including its tail."""
    stop = end + 1
    if stop - start <= 20:
        return [(start, stop)]
    starts = list(range(start, stop - 19, 10))
    tail = stop - 20
    if starts[-1] != tail:
        starts.append(tail)
    return [(first, first + 20) for first in starts]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    complete = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as handle:
            complete = {(r["clip"], r["segment"], int(r["window_start_s"]))
                        for r in csv.DictReader(handle)}
    model = BboxTubeTemporalModel.from_package(PACKAGE, device="cpu")
    with OUT.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not OUT.stat().st_size:
            writer.writeheader()
        cached_clip = None
        frames = []
        detections = {}
        detector_ms = 0.0
        for clip, segment, start, end in SEGMENTS:
            spans = [span for span in windows(start, end)
                     if (clip, segment, span[0]) not in complete]
            if not spans:
                continue
            if clip != cached_clip:
                frames = model.load_sequence(extract(clip))
                started = time.perf_counter()
                found = model.detect(frames)
                detector_ms = (time.perf_counter() - started) * 1000
                detections = {d.frame_id: d for d in found}
                cached_clip = clip
                print(f"{clip}: {len(frames)} sampled frames, detector {detector_ms:.0f}ms",
                      flush=True)
            if end >= len(frames):
                raise RuntimeError(f"Segment exceeds {clip}: {end} >= {len(frames)}")
            for first, stop in spans:
                subset = frames[first:stop]
                started = time.perf_counter()
                output = model.predict(subset, frame_detections=detections)
                temporal_ms = (time.perf_counter() - started) * 1000
                details = output.details
                kept = details.get("tubes", {}).get("kept", [])
                probabilities = [float(t["probability"]) for t in kept
                                 if t.get("probability") is not None]
                row = {
                    "clip": clip, "segment": segment,
                    "window_start_s": first, "window_end_s": stop - 1,
                    "frames": len(subset),
                    "detector_hit_frames": sum(bool(detections[f.frame_id].detections)
                                               for f in subset),
                    "candidate_tubes": details.get("tubes", {}).get("num_candidates", 0),
                    "kept_tubes": len(kept),
                    "max_tube_probability": max(probabilities, default=0.0),
                    "smoke_decision": int(output.is_positive),
                    "detector_ms_clip": round(detector_ms, 1),
                    "temporal_ms_window": round(temporal_ms, 1),
                }
                writer.writerow(row)
                handle.flush()
                print(f"  {segment} {first}-{stop - 1}s: detector hits "
                      f"{row['detector_hit_frames']}/{len(subset)}, "
                      f"tubes {len(kept)}, p={row['max_tube_probability']:.3f}, "
                      f"decision={row['smoke_decision']}", flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
