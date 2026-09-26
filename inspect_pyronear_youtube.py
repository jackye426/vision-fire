"""Save a visual audit of PyroNear's temporal companion detector and tubes."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".yolo-config"))

import cv2
from PIL import Image, ImageDraw
from temporal_model.core.model import BboxTubeTemporalModel

from benchmark_pyronear_temporal_youtube import PACKAGE, OUT, extract


REVIEW = (
    ("CI6YpclCYA4", 0, 20, (0, 5, 10, 14, 18), ((0, 20),)),
    ("WsUjSE-ibKo", 0, 30, (0, 2, 7, 10, 20), ((0, 20), (10, 30))),
    ("5NAkyEmC0IU", 20, 53, (20, 30, 35, 40, 50), ((30, 50), (33, 53))),
)


def main() -> None:
    model = BboxTubeTemporalModel.from_package(PACKAGE, device="cpu")
    audit = {}
    sheet = Image.new("RGB", (1500, 3 * 210), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (clip, lo, hi, review_seconds, windows) in enumerate(REVIEW):
        paths = extract(clip)
        frames = model.load_sequence(paths)
        detected = model.detect(frames[lo:hi])
        cache = {d.frame_id: d for d in detected}
        by_second = {lo + i: d for i, d in enumerate(detected)}
        window_data = []
        for first, stop in windows:
            output = model.predict(frames[first:stop], frame_detections=cache)
            window_data.append({"start_s": first, "end_s": stop - 1,
                                "smoke_decision": bool(output.is_positive),
                                "trigger_frame_index": output.trigger_frame_index,
                                "details": output.details})
        audit[clip] = {
            "companion": [
                {"time_s": sec, "boxes": [vars(box) for box in frame.detections]}
                for sec, frame in by_second.items() if frame.detections
            ],
            "windows": window_data,
        }
        for col, second in enumerate(review_seconds):
            raw = cv2.imread(str(paths[second]))
            height, width = raw.shape[:2]
            for box in by_second[second].detections:
                x1 = round((box.cx - box.w / 2) * width)
                y1 = round((box.cy - box.h / 2) * height)
                x2 = round((box.cx + box.w / 2) * width)
                y2 = round((box.cy + box.h / 2) * height)
                cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 220, 255), 3)
                cv2.putText(raw, f"{box.confidence:.2f}", (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            tile = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            tile.thumbnail((300, 180))
            x, y = col * 300, row_index * 210
            sheet.paste(tile, (x, y + 25))
            draw.text((x + 4, y + 4), f"{clip} {second}s ({len(by_second[second].detections)} boxes)",
                      fill="black")
    destination = OUT.parent / "pyronear_candidate_review.jpg"
    sheet.save(destination, quality=90)
    details_path = OUT.parent / "pyronear_candidate_details.json"
    details_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Saved {destination} and {details_path}")


if __name__ == "__main__":
    main()
