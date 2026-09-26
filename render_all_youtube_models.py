"""Render browser-playable videos of every tested model on the user clips.

Source playback is sampled at 5 fps. The five frame-detector predictions were
computed at 1 fps and are held until the next sampled second; the header states
this. The PyroNear temporal panel shows companion YOLO proposals and the last
completed 20-frame window verdict. It is never a per-frame temporal verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from benchmark_user_youtube import CLIPS, MEDIA, MODELS, OUT


OUTPUT = OUT.parent
TEMP = OUTPUT / "render_tmp"
SAMPLE_FPS = 5
SCORE = 0.25
HEADER = 70
FOOTER = 28
TILE = (960, 540)
NAMES = {
    "dfire_yolov8n": "D-Fire YOLOv8n  |  smoke + flame",
    "dfine": "FireViewer D-FINE M  |  smoke + flame",
    "yolo": "FireViewer YOLO11-M  |  smoke + flame",
    "soul": "SoulPerforms YOLOv8n  |  flame only",
    "pyronear": "PyroNear Sensitive  |  smoke only",
    "pyronear_temporal": "PyroNear temporal pipeline  |  smoke",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def text_line(frame: np.ndarray, value: str, position: tuple[int, int],
              size: float = 0.63, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, value, position, cv2.FONT_HERSHEY_SIMPLEX, size,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, value, position, cv2.FONT_HERSHEY_SIMPLEX, size,
                color, 2, cv2.LINE_AA)


def draw_boxes(frame: np.ndarray, boxes: list[dict]) -> tuple[int, int]:
    smoke = flame = 0
    height, width = frame.shape[:2]
    for detection in boxes:
        if float(detection["score"]) < SCORE:
            continue
        label = detection["label"]
        color = (60, 185, 255) if label == "flame" else (60, 225, 100)
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text_line(frame, f"{label} {float(detection['score']):.2f}",
                  (x1, max(22, y1 - 7)), 0.58, color)
        smoke += label == "smoke"
        flame += label == "flame"
    return smoke, flame


def draw_companion(frame: np.ndarray, boxes: list[dict]) -> int:
    height, width = frame.shape[:2]
    for detection in boxes:
        cx, cy = float(detection["cx"]), float(detection["cy"])
        bw, bh = float(detection["w"]), float(detection["h"])
        x1 = max(0, min(width - 1, round((cx - bw / 2) * width)))
        y1 = max(0, min(height - 1, round((cy - bh / 2) * height)))
        x2 = max(0, min(width - 1, round((cx + bw / 2) * width)))
        y2 = max(0, min(height - 1, round((cy + bh / 2) * height)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), 3)
        text_line(frame, f"proposal {float(detection['confidence']):.2f}",
                  (x1, max(22, y1 - 7)), 0.58, (0, 230, 255))
    return len(boxes)


def temporal_status(clip: str, second: int, windows: list[dict]) -> str:
    if clip == "ZBrOzQRoI-E":
        return "not run: edited compilation"
    segment = (
        "angle1" if second <= 52 else "angle2"
    ) if clip == "5NAkyEmC0IU" else {
        "CI6YpclCYA4": "battery_fire",
        "WsUjSE-ibKo": "recycling",
        "K3ML54RtbAo": "screen_recording",
        "wm22sUiq9fE": "screen_recording",
    }[clip]
    finished = [r for r in windows if r["clip"] == clip and r["segment"] == segment
                and int(r["window_end_s"]) <= second]
    if not finished:
        return "waiting for first 20-frame window"
    latest = max(finished, key=lambda r: int(r["window_end_s"]))
    verdict = "SMOKE" if int(latest["smoke_decision"]) else "clear"
    return (f"last completed window {latest['window_start_s']}-{latest['window_end_s']}s: "
            f"{verdict}  p={float(latest['max_tube_probability']):.2f}")


def panel(source: np.ndarray, clip: str, model: str, second: int,
          detector_rows: dict, companion_rows: dict, windows: list[dict]) -> np.ndarray:
    picture = source.copy()
    if model == "pyronear_temporal":
        proposal_count = draw_companion(picture, companion_rows.get((clip, second), []))
        status = temporal_status(clip, second, windows)
        heading = NAMES[model]
        subheading = status
        footer = f"{second}s sample | {proposal_count} yellow proposals; 20s window verdict"
    else:
        boxes = detector_rows[(clip, model, second)]
        smoke, flame = draw_boxes(picture, boxes)
        heading = NAMES[model]
        subheading = f"smoke {smoke}  flame {flame}  |  output score >= {SCORE:.2f}"
        footer = f"{second}s inference sample held to next second | 5 fps playback"
    height, width = picture.shape[:2]
    canvas = np.full((height + HEADER + FOOTER, width, 3), (24, 24, 24), np.uint8)
    canvas[HEADER:HEADER + height] = picture
    text_line(canvas, heading, (12, 28), 0.64)
    text_line(canvas, subheading, (12, 58), 0.54, (180, 230, 255))
    text_line(canvas, footer, (12, height + HEADER + 21), 0.54,
              (190, 220, 245))
    return canvas


def tile(panel_image: np.ndarray) -> np.ndarray:
    width, height = TILE
    out = np.full((height, width, 3), (12, 12, 12), np.uint8)
    src_h, src_w = panel_image.shape[:2]
    ratio = min(width / src_w, height / src_h)
    resized = cv2.resize(panel_image, (round(src_w * ratio), round(src_h * ratio)),
                         interpolation=cv2.INTER_AREA)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    out[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return out


def writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             SAMPLE_FPS, size)
    if not result.isOpened():
        raise RuntimeError(f"Cannot write {path}")
    return result


def transcode(source: Path, destination: Path) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-an", str(destination)], check=True)
    source.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="append", choices=CLIPS,
                        help="Render only this clip; repeat for more clips")
    args = parser.parse_args()
    TEMP.mkdir(exist_ok=True)
    detector_rows = {(r["clip"], r["model"], int(r["time_s"])): json.loads(r["detections"])
                     for r in read_csv(OUT)}
    companion_path = OUTPUT / "pyronear_companion_frames.csv"
    companion_rows = ({(r["clip"], int(r["time_s"])): json.loads(r["detections"])
                       for r in read_csv(companion_path)} if companion_path.exists() else {})
    windows_path = OUTPUT / "pyronear_temporal_windows.csv"
    windows = read_csv(windows_path) if windows_path.exists() else []
    all_models = (*MODELS, "pyronear_temporal")
    for clip in (args.clip or CLIPS):
        cap = cv2.VideoCapture(str(MEDIA / f"{clip}.mp4"))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {clip}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_to_write = int((source_frames - 1) / source_fps * SAMPLE_FPS) + 1
        if not all((clip, model, second) in detector_rows for model in MODELS
                   for second in range(int((source_frames - 1) / source_fps) + 1)):
            raise RuntimeError(f"Missing 1 fps model results for {clip}")
        individual = {model: TEMP / f"{clip}_{model}.mp4" for model in all_models}
        grid_path = TEMP / f"{clip}_all_models.mp4"
        individual_writers = {model: writer(path, (width, height + HEADER + FOOTER))
                              for model, path in individual.items()}
        grid_writer = writer(grid_path, (TILE[0] * 3, TILE[1] * 2))
        try:
            for sample in range(frames_to_write):
                source_index = min(round(sample * source_fps / SAMPLE_FPS), source_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {clip} frame {source_index}")
                second = sample // SAMPLE_FPS
                panels = []
                for model in all_models:
                    rendered = panel(bgr, clip, model, second,
                                     detector_rows, companion_rows, windows)
                    individual_writers[model].write(rendered)
                    panels.append(tile(rendered))
                grid = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
                grid_writer.write(grid)
                if sample % 100 == 0:
                    print(f"{clip}: rendered {sample}/{frames_to_write} frames", flush=True)
        finally:
            cap.release()
            grid_writer.release()
            for stream in individual_writers.values():
                stream.release()
        files = [(grid_path, OUTPUT / f"{clip}_all_models_1fps_boxes_5fps_h264.mp4")]
        files += [(path, OUTPUT / f"{clip}_{model}_1fps_boxes_5fps_h264.mp4")
                  for model, path in individual.items()]
        for source, destination in files:
            transcode(source, destination)
        print(f"{clip}: saved comparison and {len(all_models)} individual videos", flush=True)


if __name__ == "__main__":
    main()
