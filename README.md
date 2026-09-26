# Smoke & fire model lab

A local, CPU-based comparison of five existing vision checkpoints. Upload an image or video, adjust the score threshold, and inspect bounding boxes, first raw detections, persistent alerts, and CPU inference time. Video is sampled at a configurable interval, so activity between frames can be missed. No footage leaves this machine during inference.

Read [MODEL_GUIDE.md](MODEL_GUIDE.md) for the types of model, setup tradeoffs, rights, published performance, and a fair trial method.

Read [TEMPORAL_EXPERIMENT.md](TEMPORAL_EXPERIMENT.md) for the staged comparison of repeated-frame YOLO, two specialists, detector plus temporal verification, PyroNear's image-encoder/temporal-head system, full-video backbones, and vision-language models. It includes the new PyroNear temporal run and the specialist-pair result.

Read [INDEPENDENT_DATASETS.md](INDEPENDENT_DATASETS.md) for public datasets beyond D-Fire and an annotated video of D-Fire YOLOv8n running on an external vehicle-fire clip. The app embeds that video and can run the five detectors interactively on its original footage.

The [cross-dataset report](CROSS_DATASET_REPORT.md) contains the first measured outside-D-Fire comparison: five FURG videos with flame rectangles and six CMU industrial-smoke clips with clip-level labels. It includes the released PyroNear temporal smoke pipeline on the CMU clips.

The [Roboflow flame-positive report](outputs/roboflow_flame_positive/REPORT.md) compares four flame-capable checkpoints on 413 source-ID-filtered annotated test images. The [user-selected YouTube clip report](outputs/youtube/REPORT.md) runs all five frame detectors on four retrievable fire videos at 1 fps, adds PyroNear's released temporal pipeline on three of them, checks candidate locations visually, and links two annotated D-Fire video renders. Both are exploratory transfer checks; the YouTube clips do not have frame-level ground truth.

[Watch all model recordings](WATCH_MODEL_VIDEOS.md) lists side-by-side and individual model videos for each retrievable user clip. The rendered videos are local generated files and are not included in Git; use the benchmark and rendering scripts with the source clips to recreate them.

The [two supplied screen recordings report](SUPPLIED_CLIP_REPORT.md) covers the previously inaccessible K3ML and wm22 videos, including their first visible flame boxes, misleading loader/wall detections, and the temporal pipeline's window decisions.

The [D-Fire benchmark report](BENCHMARK_REPORT.md) contains the first measured comparison on **160 labeled test images and six test videos**, plus a smoke-specialist/fire-specialist OR result and ten windows from the released PyroNear temporal smoke system. The app opens with its results table and lets you select an original test image or clip to inspect the boxes. Compact result tables under `outputs/` are included in Git; downloaded media under `dataset/`, model weights under `models/`, and rendered videos and images under `outputs/` stay local.

The [EMR Tilbury visit notes](https://app.notion.com/p/3e5196f5637381aea9c8e09aeb82013a) motivate the experiment: existing CCTV and thermal feeds exceed the attention of human monitors; the useful first measure is how many actionable minutes earlier smoke can be surfaced without swamping reviewers. The overnight East Tilbury incident happened at a different site from the Tilbury Dock visit. Its approximate timestamps should not be treated as ground truth without original footage.

## Run

In PowerShell from this folder:

```powershell
python -m pip install -r requirements.txt
python download_models.py
python -m streamlit run app.py
```

The first download needs internet access and stores the ready-made ONNX files under `models/`. These files are ignored by Git. The app runs inference locally and has no camera, alarm, or network integration. Tested with Python 3.14 and CPU ONNX Runtime.

The D-Fire YOLOv8n source is a PyTorch checkpoint without an ONNX export. To add that model to the local app, run these extra commands once:

```powershell
python -m venv .venv-export
.\.venv-export\Scripts\python.exe -m pip install -r requirements-export.txt
.\.venv-export\Scripts\python.exe export_dfire.py
```

The exporter pins and checks the checkpoint, then writes `models/dfire_yolov8n/best.onnx`. Only the exporter loads the `.pt` file; the main app loads ONNX. The export environment is isolated and ignored by Git.

## Reproduce the D-Fire trial

The official image archive is about 2.83 GB. The download scripts store it under `dataset/`, and the fixed sample and result CSVs under `outputs/dfire_benchmark/`.

```powershell
python -m pip install -r requirements-benchmark.txt
python -m playwright install chromium
python fetch_dfire_splits.py
python fetch_dfire_archive.py
python fetch_dfire_videos.py
python benchmark_dfire_images.py
python benchmark_dfire_videos.py
python benchmark_dfire_localization.py
python summarize_dfire.py
python summarize_specialist_pair.py
python sweep_alert_rules.py
```

The archive downloader resumes partial transfers. Result CSVs resume missing model/image or model/frame rows. For a new model revision or changed sample, start a new benchmark output directory so old results are not mixed with new inference.

The optional [PyroNear temporal model](https://huggingface.co/pyronear/temporal-model) uses an isolated Python 3.12/Torch environment rather than the app's ONNX Runtime. Its released package is pinned and checked by `fetch_pyronear_temporal.py`; run `benchmark_pyronear_temporal.py` in `.venv-temporal` after installing PyroNear's `temporal-model-core[torch]`. The exact source version, result protocol and limits are in [TEMPORAL_EXPERIMENT.md](TEMPORAL_EXPERIMENT.md). The app displays its measured windows for the selected D-Fire clip, but interactive "Run comparison" still runs the five frame detectors.

## Models

| Model | Visible classes | Why include it | Source and terms |
| --- | --- | --- | --- |
| [D-Fire YOLOv8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) | Smoke, fire | Small, popular two-class baseline trained on D-Fire | AGPL-3.0; local export required |
| [SoulPerforms YOLOv8n demo](https://huggingface.co/spaces/SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio) | Fire only | Browser demo and local ONNX; illustrates why the actual label map matters | Space config says MIT; ONNX metadata says AGPL-3.0 |
| [FireViewer D-FINE M strict v1](https://huggingface.co/fireviewer/fire-smoke-dfine-m-strict-v1) | Smoke, flame | Two-class primary research candidate with a documented independent image evaluation | Model card and rights files; source-specific training-data terms |
| [FireViewer YOLO11-M strict v1](https://huggingface.co/fireviewer/fire-smoke-yolo11m-strict-v1) | Smoke, flame | Same corpus family, different architecture; published as a weak baseline | AGPL-3.0 and source-specific training-data terms |
| [Pyronear Sensitive Detector v1.1.0](https://huggingface.co/pyronear/yolo11s_sensitive-detector) | Smoke only | Early wildfire smoke specialist; useful domain-transfer comparison | Apache-2.0 per model card |

The UI normalizes source labels `fire` and `flame_visible` to **flame** for one comparison column. Model outputs and thresholds are **not calibrated fire probabilities**. Published image scores are from different datasets and do not predict EMR camera performance. Pyronear is trained for wildfire smoke, not industrial yards. The FireViewer YOLO card reports weak flame performance and a high negative-image false-alarm rate on its reported holdout. This is a benchmark candidate, not an alarm product.

## How to judge them

1. Start with the included D-Fire example collage to verify the pipeline. It already contains drawn boxes and is not a valid benchmark. Source: [D-Fire repository](https://github.com/gaia-solutions-on-demand/DFireDataset), CC0.
2. Try clear yard footage and hard negatives: dust, steam, hot work, bright lights, sun glare, exhaust, and machinery.
3. Try independent fire footage, especially the earliest visible smoke and night scenes. Compare the first raw hit with the first persistent alert, plus the number of flagged negative frames. Enter a human escalation time if known.
4. For a real retrospective EMR evaluation, use authorised historical footage and ordinary overnight clips. Record visible-smoke onset, first actionable model alert, human escalation, misses, and false alerts per camera-hour. Respect EMR's footage-sharing restrictions.

These checkpoints consume ordinary RGB images. A false-colour thermal frame is not equivalent to a temperature measurement; that needs a separate, calibrated thermal-data path. Never let this prototype automatically trigger emergency action.
