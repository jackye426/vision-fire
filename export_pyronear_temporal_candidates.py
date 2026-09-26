"""Export the released temporal pipeline's companion YOLO boxes for video review.

These boxes are proposals, not the temporal classifier's final decisions.
The final whole-window decisions are in pyronear_temporal_windows.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark_pyronear_temporal_youtube import PACKAGE, OUT, extract
from temporal_model.core.model import BboxTubeTemporalModel


CLIPS = ("CI6YpclCYA4", "WsUjSE-ibKo", "5NAkyEmC0IU",
         "K3ML54RtbAo", "wm22sUiq9fE")
DESTINATION = OUT.parent / "pyronear_companion_frames.csv"


def main() -> None:
    complete = set()
    if DESTINATION.exists():
        with DESTINATION.open(newline="", encoding="utf-8") as handle:
            complete = {r["clip"] for r in csv.DictReader(handle)}
    model = BboxTubeTemporalModel.from_package(PACKAGE, device="cpu")
    with DESTINATION.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip", "time_s", "detections"))
        if not DESTINATION.stat().st_size:
            writer.writeheader()
        for clip in CLIPS:
            if clip in complete:
                print(f"Skipping {clip}: results already present", flush=True)
                continue
            paths = extract(clip)
            frames = model.load_sequence(paths)
            detections = model.detect(frames)
            if len(detections) != len(frames):
                raise RuntimeError(f"Incomplete companion detections for {clip}")
            for second, frame in enumerate(detections):
                writer.writerow({
                    "clip": clip, "time_s": second,
                    "detections": json.dumps([vars(box) for box in frame.detections]),
                })
            handle.flush()
            print(f"Saved {len(detections)} companion frames for {clip}", flush=True)
    print(f"Saved {DESTINATION}")


if __name__ == "__main__":
    main()
