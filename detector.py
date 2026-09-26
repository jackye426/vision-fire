"""Local ONNX inference and common bounding-box output for three public detectors."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent / "models"
SPECS = {
    "dfine": {"title": "D-FINE · smoke + flame", "size": 704, "path": ROOT / "dfine" / "model.onnx", "labels": ("smoke", "flame")},
    "yolo": {"title": "YOLO11-M · smoke + flame", "size": 960, "path": ROOT / "yolo" / "model.onnx", "labels": ("smoke", "flame")},
    "dfire_yolov8n": {"title": "D-Fire YOLOv8n · smoke + fire", "size": 640, "path": ROOT / "dfire_yolov8n" / "best.onnx", "labels": ("smoke", "flame")},
    "soul": {"title": "SoulPerforms YOLOv8n · fire only", "size": 640, "path": ROOT / "soul" / "best.onnx", "labels": ("flame",)},
    "pyronear": {"title": "Pyronear · wildfire smoke", "size": 1024, "path": ROOT / "pyronear" / "best.onnx", "labels": ("smoke",)},
}


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


@lru_cache(maxsize=5)
def session(name: str):
    path = SPECS[name]["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python download_models.py")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _letterbox(rgb: np.ndarray, size: int):
    height, width = rgb.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, ratio, left, top


def _iou(box: np.ndarray, others: np.ndarray):
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    a = np.maximum(box[2] - box[0], 0) * np.maximum(box[3] - box[1], 0)
    b = np.maximum(others[:, 2] - others[:, 0], 0) * np.maximum(others[:, 3] - others[:, 1], 0)
    return inter / np.maximum(a + b - inter, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold=0.5):
    keep = []
    for class_id in np.unique(classes):
        indices = np.where(classes == class_id)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]
        while len(indices) and len(keep) < 300:
            chosen = indices[0]
            keep.append(chosen)
            indices = indices[1:]
            if len(indices):
                indices = indices[_iou(boxes[chosen], boxes[indices]) <= iou_threshold]
    return keep


def detect(name: str, image: Image.Image, floor: float = 0.05) -> list[Detection]:
    """Return candidate detections above a low floor; UI thresholding happens later."""
    spec = SPECS[name]
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    model = session(name)
    if name == "dfine":
        resized = cv2.resize(rgb, (704, 704), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        labels, boxes, scores = model.run(None, {
            "images": tensor,
            # This ONNX graph scales x by the first size and y by the second.
            # Its published example passes (height, width), which misplaces
            # boxes on non-square images; labeled D-Fire boxes verify (w, h).
            "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
        })
        records = []
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            class_id = int(label)
            if class_id not in (0, 1) or float(score) < floor:
                continue
            x1, y1, x2, y2 = box
            records.append(Detection(spec["labels"][class_id], float(score),
                                     (int(x1), int(y1), int(x2), int(y2))))
        return sorted(records, key=lambda d: d.score, reverse=True)

    size = spec["size"]
    padded, ratio, left, top = _letterbox(rgb, size)
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    output = model.run(None, {model.get_inputs()[0].name: tensor})[0]
    predictions = output[0].T  # anchors x (xywh + one score per class)
    class_scores = predictions[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    chosen = scores >= floor
    predictions, classes, scores = predictions[chosen], classes[chosen], scores[chosen]
    if len(predictions) == 0:
        return []
    cx, cy, bw, bh = predictions[:, :4].T
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = _nms(boxes, scores, classes)
    boxes = boxes[keep]
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - left) / ratio
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - top) / ratio
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, width)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, height)
    records = [Detection(spec["labels"][int(classes[i])], float(scores[i]), tuple(map(int, box)))
               for box, i in zip(boxes, keep)]
    return sorted(records, key=lambda d: d.score, reverse=True)


def annotate(image: Image.Image, detections: list[Detection], threshold: float) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for detection in detections:
        if detection.score < threshold:
            continue
        color = (60, 170, 255) if detection.label == "flame" else (40, 220, 130)
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{detection.label} {detection.score:.2f}", (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
