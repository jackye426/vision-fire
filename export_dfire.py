"""Download a pinned D-Fire YOLOv8n .pt and export it to ONNX once.

Run in a separate exporter environment; the main app only loads the ONNX file.
"""

import hashlib
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "dfire_yolov8n"
REPO = "rabahdev/fire-smoke-yolov8n"
REVISION = "13017fe8af477c25f5298d168e2dfede4b000753"
SOURCE_SHA256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config_dir = ROOT / ".yolo-config"
    config_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(repo_id=REPO, filename="best.pt", revision=REVISION,
                                  local_dir=str(MODEL_DIR)))
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Downloaded D-Fire checkpoint did not match expected SHA-256")
    # The .pt format uses Python pickle. Load only this pinned, hash-checked checkpoint.
    from ultralytics import YOLO
    model = YOLO(str(source))
    if model.names != {0: "smoke", 1: "fire"}:
        raise ValueError(f"Unexpected checkpoint classes: {model.names}")
    output = Path(model.export(format="onnx", imgsz=640, opset=12,
                               simplify=False, dynamic=False, device="cpu"))
    print(f"ONNX model: {output} ({sha256(output)})")


if __name__ == "__main__":
    main()
