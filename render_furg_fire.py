"""Render D-Fire YOLOv8n over an independent FURG vehicle-fire video.

This is a visual case study, not a model benchmark. The FURG XML supplies
frame-level flame rectangles. Output is sampled at every third source frame.
"""

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
from PIL import Image

from detector import detect


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset" / "furg" / "hand_held_camera_wildfire.mp4"
LABELS = ROOT / "dataset" / "furg" / "hand_held_camera_wildfire.xml"
OUT = ROOT / "outputs" / "furg"
VIDEO = OUT / "hand_held_camera_wildfire_dfire_yolov8n.mp4"
CSV = OUT / "hand_held_camera_wildfire_dfire_yolov8n.csv"
THRESHOLD = 0.25
STRIDE = 3


def annotations() -> dict[int, list[tuple[int, int, int, int]]]:
    root = ET.parse(LABELS).getroot()
    boxes = {}
    for item in root.find("frames"):
        number = int(item.findtext("frameNumber"))
        boxes[number] = []
        for child in item.find("annotations"):
            values = list(map(int, child.text.split()))
            if len(values) == 4:
                x, y, w, h = values
                boxes[number].append((x, y, x + w, y + h))
    return boxes


def main() -> None:
    if not SOURCE.exists() or not LABELS.exists():
        raise FileNotFoundError("Run python fetch_furg_examples.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    gt = annotations()
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"Could not read {SOURCE}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"),
                             source_fps / STRIDE, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated MP4")
    rows = []
    frame_number = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_number % STRIDE == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            predictions = [d for d in detect("dfire_yolov8n", Image.fromarray(rgb))
                           if d.score >= THRESHOLD]
            for d in predictions:
                color = (0, 150, 255) if d.label == "flame" else (0, 220, 100)
                x1, y1, x2, y2 = d.box
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
                cv2.putText(bgr, f"{d.label} {d.score:.2f}",
                            (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, color, 2, cv2.LINE_AA)
            time_s = frame_number / source_fps
            cv2.rectangle(bgr, (0, 0), (width, 44), (25, 25, 25), -1)
            cv2.putText(bgr, f"D-Fire YOLOv8n  |  {time_s:05.1f}s  |  score >= {THRESHOLD:.2f}",
                        (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(bgr)
            rows.append({
                "source_frame": frame_number, "time_s": round(time_s, 3),
                "furg_flame_labeled": int(bool(gt.get(frame_number))),
                "model_smoke": int(any(d.label == "smoke" for d in predictions)),
                "model_flame": int(any(d.label == "flame" for d in predictions)),
                "max_smoke_score": max((round(d.score, 4) for d in predictions
                                        if d.label == "smoke"), default=0),
                "max_flame_score": max((round(d.score, 4) for d in predictions
                                        if d.label == "flame"), default=0),
                "model_boxes": len(predictions),
            })
        frame_number += 1
    cap.release()
    writer.release()
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    first_gt = next((r["time_s"] for r in rows if r["furg_flame_labeled"]), None)
    first_model = next((r["time_s"] for r in rows if r["model_flame"]), None)
    labeled = [r for r in rows if r["furg_flame_labeled"]]
    hits = sum(r["model_flame"] for r in labeled)
    print(f"Rendered {len(rows)} frames from {frame_number} source frames to {VIDEO}")
    print(f"First FURG flame label: {first_gt}s; first YOLO flame box: {first_model}s")
    print(f"YOLO flame box on {hits}/{len(labeled)} sampled flame-labeled frames")
    print(f"CSV: {CSV}")


if __name__ == "__main__":
    main()
