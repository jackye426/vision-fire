"""Download pinned, public ONNX checkpoints. No remote Python is executed."""

from pathlib import Path
import hashlib
import shutil
import tarfile

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent / "models"

MODELS = {
    "dfine": {
        "repo": "fireviewer/fire-smoke-dfine-m-strict-v1",
        "revision": "0ef733968748c7ac172273db8c381ea5ea5ff046",
        "filename": "model.onnx",
        "sha256": "d4ad38e8c09756b975b177faa8440dc1ed9f52df10828c9981484e9d1e1db2ee",
    },
    "yolo": {
        "repo": "fireviewer/fire-smoke-yolo11m-strict-v1",
        "revision": "a75aa44716658f02eb85b4e7bba6a358040e6f86",
        "filename": "model.onnx",
        "sha256": "2532e15c16cef8909317ddcf5e614dce2eb420319d40199b2980b429925ba129",
    },
    "soul": {
        "repo": "SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio",
        "repo_type": "space",
        "revision": "b749972afbc68f365f9769b162de997f502a8910",
        "filename": "best.onnx",
        "sha256": "3316a31ab40aa168f455e5f50a86d97db346d1d3490208754e06cfe2409fcd7c",
    },
    "pyronear": {
        "repo": "pyronear/yolo11s_sensitive-detector",
        "revision": "v1.1.0",
        "filename": "onnx_cpu.tar.gz",
        "sha256": "cb6402cdea164ef7a4082fc726972d1cc7380239ca13d5e8adec70010991e888",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(selected=None):
    selected = selected or MODELS.keys()
    paths = {}
    for name in selected:
        info = MODELS[name]
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / ("best.onnx" if name in ("pyronear", "soul") else "model.onnx")
        if not target.exists():
            source = Path(hf_hub_download(
                repo_id=info["repo"], repo_type=info.get("repo_type", "model"), filename=info["filename"],
                revision=info["revision"], local_dir=str(folder),
            ))
            if name == "pyronear":
                with tarfile.open(source, "r:gz") as archive:
                    member = archive.getmember("./best.onnx")
                    with archive.extractfile(member) as input_file, target.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
            else:
                target = source
        if info.get("sha256") and sha256(target) != info["sha256"]:
            raise ValueError(f"Unexpected SHA-256 for {name}: {target}")
        paths[name] = target
        print(f"{name}: {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return paths


if __name__ == "__main__":
    download()
