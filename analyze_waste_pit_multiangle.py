"""Exploratory display of two edited waste-pit views.

The user identified a dozer landmark at video second 2 in angle 1 and about
second 56 in angle 2, then observed the second view lagged by about one second.
The two sections also play at different speeds. Two user-identified matching
events give a visual time warp: grabber drop 7 -> 59 s and water cannon 31 ->
71 s. It maps the earlier dozer estimate 2 -> 56.5 s. This is not capture-clock
synchronization; no cross-view alert timing is measured.
Detections come from the existing 1 fps frame_results.csv; no model is rerun.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CLIP = "5NAkyEmC0IU"
SOURCE = ROOT / "dataset" / "youtube" / f"{CLIP}.mp4"
INPUT = ROOT / "outputs" / "youtube" / "frame_results.csv"
OUT = ROOT / "outputs" / "youtube" / "multiangle_waste_pit.csv"
SHEET = ROOT / "outputs" / "youtube" / f"{CLIP}_aligned_views.jpg"
VIDEO = ROOT / "outputs" / "youtube" / f"{CLIP}_aligned_dfine_h264.mp4"
MODELS = ("dfire_yolov8n", "dfine", "yolo")
ANCHOR_A, ANCHOR_B = 2, 56.5
SECOND_A, SECOND_B = 7, 59
THIRD_A, THIRD_B = 31, 71
LAST_COMMON_SECOND = 45  # Display index; angle 1 source 2..47 in this case study.
SCORE = 0.25
IOU = 0.20

# Box-centre regions around the visually reviewed pit fire/plume. Chosen after
# seeing this positive clip: these are an illustrative check, not a deployable
# region detector or independent test threshold.
REGIONS = {
    ("angle1", "flame"): (500, 500, 800, 710),
    ("angle1", "smoke"): (500, 150, 950, 700),
    ("angle2", "flame"): (600, 300, 1100, 600),
    ("angle2", "smoke"): (600, 150, 1100, 600),
}


def angle2_video_time(angle1_video_time: float) -> float:
    """Approximate angle-2 source time using two matching physical events."""
    return SECOND_B + (angle1_video_time - SECOND_A) * (
        (THIRD_B - SECOND_B) / (THIRD_A - SECOND_A))


def angle1_video_time(angle2_video_time: float) -> float:
    """Map an angle-2 source time onto the shared angle-1 playback clock."""
    return SECOND_A + (angle2_video_time - SECOND_B) * (
        (THIRD_A - SECOND_A) / (THIRD_B - SECOND_B))


def load_rows() -> dict[tuple[str, int], list[dict]]:
    found = {}
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != CLIP or row["model"] not in MODELS:
                continue
            key = row["model"], int(row["time_s"])
            if key in found:
                raise ValueError(f"Duplicate prediction: {key}")
            found[key] = json.loads(row["detections"])
    for model in MODELS:
        for second in (*range(ANCHOR_A, 53), *range(int(ANCHOR_B), 103)):
            if (model, second) not in found:
                raise ValueError(f"Missing prediction: {(model, second)}")
    return found


def boxes(data: dict, model: str, second: int, label: str,
          angle: str, focused: bool) -> list[list[int]]:
    area = REGIONS[(angle, label)]
    result = []
    for item in data[(model, second)]:
        if item["label"] != label or float(item["score"]) < SCORE:
            continue
        box = item["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not focused or area[0] <= cx <= area[2] and area[1] <= cy <= area[3]:
            result.append(box)
    return result


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def first_hit(data: dict, model: str, label: str, angle: str,
              focused: bool, anchor: int) -> int | None:
    for common in range(LAST_COMMON_SECOND + 1):
        if boxes(data, model, anchor + common, label, angle, focused):
            return common
    return None


def first_repeat(data: dict, model: str, label: str, angle: str,
                 focused: bool, anchor: int) -> int | None:
    for common in range(1, LAST_COMMON_SECOND + 1):
        current = boxes(data, model, anchor + common, label, angle, focused)
        previous = boxes(data, model, anchor + common - 1, label, angle, focused)
        if any(iou(a, b) >= IOU for a in current for b in previous):
            return common
    return None


def fmt(value: int | None) -> str:
    return "" if value is None else str(value)


def make_sheet() -> None:
    if not SOURCE.exists():
        print(f"Skipped visual sheet: source clip absent at {SOURCE}")
        return
    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    panels = []
    for offset in (0, 3, 5, 11, 20, 29):
        a = ANCHOR_A + offset
        b = angle2_video_time(a)
        pair = []
        for second, angle in ((a, "Angle 1"), (b, "Angle 2")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame at {second:.1f}s")
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            panel = np.full((395, 640, 3), (25, 25, 25), np.uint8)
            panel[35:] = frame
            cv2.putText(panel, f"{angle}: source {second:.1f}s  |  visual time warp",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
            pair.append(panel)
        panels.append(np.hstack(pair))
    cap.release()
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(SHEET), np.vstack(panels),
                       [cv2.IMWRITE_JPEG_QUALITY, 89]):
        raise RuntimeError(f"Cannot write {SHEET}")
    print(f"Visual alignment sheet: {SHEET}")


def draw_text(image: np.ndarray, value: str, position: tuple[int, int],
              size: float = 0.62, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, size,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, size,
                color, 2, cv2.LINE_AA)


def make_video(data: dict) -> None:
    """Render a short D-FINE replay; detections remain the saved 1 fps outputs."""
    if not SOURCE.exists():
        print(f"Skipped paired video: source clip absent at {SOURCE}")
        return
    source = cv2.VideoCapture(str(SOURCE))
    if not source.isOpened():
        raise RuntimeError(f"Cannot open {SOURCE}")
    fps = source.get(cv2.CAP_PROP_FPS)
    temp = VIDEO.with_name(VIDEO.stem + "_temp.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"),
                             5, (1280, 454))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {temp}")
    try:
        for index in range(34 * 5):
            common = index / 5
            sampled = int(common)
            canvas = np.full((454, 1280, 3), (25, 25, 25), np.uint8)
            draw_text(canvas, "APPROXIMATE 2-LANDMARK VISUAL ALIGNMENT - NOT CAMERA CLOCK SYNC",
                      (12, 28), 0.62, (190, 230, 255))
            draw_text(canvas, f"View 1 {ANCHOR_A + common:.1f}s  |  view 2 {angle2_video_time(ANCHOR_A + common):.1f}s",
                      (12, 54), 0.60)
            for angle, left in (("angle1", 0), ("angle2", 640)):
                t = (ANCHOR_A + common if angle == "angle1" else
                     angle2_video_time(ANCHOR_A + common))
                source.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
                ok, frame = source.read()
                if not ok:
                    raise RuntimeError(f"Cannot read {angle} at {t:.1f}s")
                height, width = frame.shape[:2]
                second = int(ANCHOR_A + sampled) if angle == "angle1" else int(t)
                for detection in data[("dfine", second)]:
                    if float(detection["score"]) < SCORE:
                        continue
                    label = detection["label"]
                    box = detection["box"]
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    region = REGIONS[(angle, label)]
                    focused = region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
                    color = ((55, 190, 255) if label == "flame" else (70, 230, 95)) if focused else (160, 160, 160)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                                  color, 3 if focused else 1)
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                canvas[70:430, left:left + 640] = frame
                draw_text(canvas, f"{angle}  source {t:.1f}s  |  saved boxes at {second}s",
                          (left + 8, 450), 0.52)
            writer.write(canvas)
    finally:
        writer.release()
        source.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(temp), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-an", str(VIDEO)], check=True)
    temp.unlink()
    print(f"Illustrative paired D-FINE video: {VIDEO}")


def main() -> None:
    data = load_rows()
    result = []
    for focused in (False, True):
        for label in ("flame", "smoke"):
            for model in MODELS:
                a_hit = first_hit(data, model, label, "angle1", focused, ANCHOR_A)
                a_repeat = first_repeat(data, model, label, "angle1", focused, ANCHOR_A)
                b_hit = first_hit(data, model, label, "angle2", focused, int(ANCHOR_B))
                b_repeat = first_repeat(data, model, label, "angle2", focused, int(ANCHOR_B))
                result.append({
                    "box_filter": "fire_area" if focused else "whole_frame",
                    "class": label,
                    "model": model,
                    "angle1_first_hit_source_s": fmt(None if a_hit is None else ANCHOR_A + a_hit),
                    "angle1_first_repeat_source_s": fmt(None if a_repeat is None else ANCHOR_A + a_repeat),
                    "angle2_first_hit_source_s": fmt(None if b_hit is None else int(ANCHOR_B) + b_hit),
                    "angle2_first_repeat_source_s": fmt(None if b_repeat is None else int(ANCHOR_B) + b_repeat),
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result[0])
        writer.writeheader()
        writer.writerows(result)
    print(f"Saved {OUT}")
    for row in result:
        print(row)
    make_sheet()
    make_video(data)


if __name__ == "__main__":
    main()
