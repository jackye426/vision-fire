"""Render sampled D-Fire boxes on a downloaded user clip for visual review.

Example: python render_dfire_youtube.py WsUjSE-ibKo
The output is silent and sampled at 5 frames/second; its clock is playback time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detector import annotate, detect


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", choices=("CI6YpclCYA4", "WsUjSE-ibKo", "ZBrOzQRoI-E", "5NAkyEmC0IU"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--score", type=float, default=0.25)
    args = parser.parse_args()
    if args.sample_fps <= 0 or not 0 < args.score <= 1:
        parser.error("sample-fps must be positive and score must be in (0, 1]")

    source = ROOT / "dataset" / "youtube" / f"{args.clip}.mp4"
    destination = ROOT / "outputs" / "youtube" / f"{args.clip}_dfire_5fps.mp4"
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.sample_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {destination}")
    frame_index = 0
    sample_index = 0
    try:
        while frame_index < frame_count:
            ok, bgr = cap.read()
            if not ok:
                break
            target = round(sample_index * source_fps / args.sample_fps)
            if frame_index == target:
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                found = detect("dfire_yolov8n", image, floor=args.score)
                overlay = annotate(image, found, threshold=args.score)
                result = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
                t = frame_index / source_fps
                cv2.rectangle(result, (0, 0), (width, 35), (15, 15, 15), -1)
                cv2.putText(result, f"D-Fire YOLOv8n | score >= {args.score:.2f} | playback {t:.1f}s",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                writer.write(result)
                sample_index += 1
                if sample_index % 50 == 0:
                    print(f"Rendered {sample_index} frames ({t:.1f}s)", flush=True)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    print(f"Saved {destination} ({sample_index} frames)", flush=True)


if __name__ == "__main__":
    main()
