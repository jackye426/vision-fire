# D-Fire smoke and fire model trial

Run on 25 September 2026. This is an exploratory comparison of existing RGB detectors for the [EMR monitoring question](https://app.notion.com/p/3e5196f5637381aea9c8e09aeb82013a): can visible smoke be surfaced early enough for human review without overwhelming the monitors?

## What was tested

The media came from the [official D-Fire repository](https://github.com/gaia-solutions-on-demand/DFireDataset): its [image archive](https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw), [published image and video splits](https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN), and [surveillance video folder](https://1drv.ms/f/c/c0bd25b6b048b01d/EhT2Jy6L-YlGvZv-gXH2SnYBENQsnUW96LpZtv_6PngjYQ). The archive has 4,306 test images and 17,221 training images; their published filename lists have no overlap. Its SHA-256 is `3824fb3ce32cfa8b538792dfd460603648d271072ac6ae34f1af0c713f60260c` (3,036,222,313 bytes).

From the test images I used a fixed random seed, `20260925`, to take 40 each with no annotation, smoke only, flame only, and both: **160 original images**. The exact names and per-image hashes are in [image_manifest.csv](outputs/dfire_benchmark/image_manifest.csv). The mix is 57 AoF camera images, 89 web images, and 14 PublicDataset images. The balanced design exposes both classes and negatives; it does **not** represent real camera prevalence. It contains sequences and related views, so 160 images are not 160 independent incidents.

Every model ran locally on CPU through ONNX Runtime with four inference threads, at its published input size. The measured median includes preprocessing, inference and box postprocessing but excludes file loading and first model initialization. A shared score cutoff of 0.25 was used; 0.50 is shown as a sensitivity check. These scores are model outputs, **not calibrated probabilities**.

Five downloadable checkpoints ran on every sampled image and clip:

| Checkpoint | Setup | Input | What it can report |
| --- | --- | ---: | --- |
| [D-Fire YOLOv8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) | Small single-frame detector, trained on D-Fire | 640 px | Smoke, flame |
| [FireViewer D-FINE M](https://huggingface.co/fireviewer/fire-smoke-dfine-m-strict-v1) | Larger transformer-style single-frame detector | 704 px | Smoke, flame |
| [FireViewer YOLO11-M](https://huggingface.co/fireviewer/fire-smoke-yolo11m-strict-v1) | Larger YOLO single-frame detector | 960 px | Smoke, flame |
| [SoulPerforms demo weights](https://huggingface.co/spaces/SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio) | Small single-frame YOLO; actual weight labels contain only `fire` | 640 px | Flame only |
| [PyroNear Sensitive Detector](https://huggingface.co/pyronear/yolo11s_sensitive-detector) | Distant wildfire smoke specialist | 1024 px | Smoke only |

The [pedbrgs research system](https://github.com/pedbrgs/Fire-Detection) is a two-stage YOLOv5 plus area-variation or temporal-persistence pipeline. Its full Docker/weights pipeline was not included in this common ONNX run. The lab's two-consecutive-hit setting illustrates a simple temporal gate, not the paper's AVT or TPT implementation. The [Roboflow community model](https://universe.roboflow.com/s-workspace-173hq/fire-smoke-detection-avnev-goqj2) is hosted; it was not batch tested locally because downloadable weights and API access were not established. Their published metrics cannot be placed in this table.

## Image results at score 0.25

“Hit” means at least one prediction of the correct class anywhere in an image with that D-Fire class. “Aligned” requires a same-class predicted box with **IoU ≥ 0.50** against at least one labeled box. Each class has 80 positive images. “Clear flagged” counts any smoke or flame output on 40 images with no D-Fire label.

| Model | Smoke hit | Smoke aligned | Flame hit | Flame aligned | Clear flagged | Median CPU ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-FINE M | 76/80 (95%) | 68/80 (85%) | 79/80 (99%) | 73/80 (91%) | **25/40** | 385 |
| D-Fire YOLOv8n | 71/80 (89%) | 64/80 (80%) | 74/80 (93%) | 67/80 (84%) | **0/40** | 53 |
| FireViewer YOLO11-M | 58/80 (73%) | 50/80 (63%) | 56/80 (70%) | 52/80 (65%) | **6/40** | 713 |
| SoulPerforms YOLOv8n | — | — | 52/80 (65%) | 36/80 (45%) | **4/40** | 54 |
| PyroNear | 19/80 (24%) | 8/80 (10%) | — | — | **2/40** | 319 |
| PyroNear smoke OR SoulPerforms flame | 19/80 (24%) | 8/80 (10%) | 52/80 (65%) | 36/80 (45%) | **6/40** | 373* |

*The pair is a decision-level OR of two independently run specialist models; its time is the median of their per-image CPU times added together. It is not a trained fusion model. [Combination scores](outputs/dfire_benchmark/specialist_pair_image.csv) were calculated from the same image results.

At score **0.50**, D-FINE's clear flags drop from 25/40 to **7/40**, while its smoke hits fall to **67/80** and flame hits to **71/80**. D-Fire YOLOv8n remains at **0/40** clear flags, but its smoke hits fall to **58/80** and flame hits to **64/80**. FireViewer YOLO11-M becomes 41/80 smoke, 43/80 flame and 2/40 clear flags. SoulPerforms becomes 39/80 flame and 3/40 clear flags; PyroNear becomes 10/80 smoke and 0/40 clear flags. Box alignment was calculated at 0.25 only.

The D-Fire model's strong result is on **the dataset family it was trained for**. It is a useful local baseline, but this trial does not show how it transfers to EMR night CCTV. Conversely, PyroNear's low D-Fire image recall does not settle its performance on its intended wildfire-camera domain. D-FINE captures more labeled events here but has a heavy clear-image alert burden at 0.25. The shared threshold is not an equal operating point across models.

The full [per-image scores](outputs/dfire_benchmark/image_results.csv), [box overlap results](outputs/dfire_benchmark/localization_results.csv), and [class confusion summary](outputs/dfire_benchmark/image_summary.csv) make these counts auditable. The official annotations can be incomplete or visually ambiguous, especially for faint smoke; “clear” here means **no D-Fire annotation**, not guaranteed absence of every possible smoke-like feature. Presence and one-box overlap are simpler than COCO mAP and should not be described as mAP.

### D-FINE box integration check

The publisher's [ONNX inference example](https://huggingface.co/fireviewer/fire-smoke-dfine-m-strict-v1/blob/main/inference.py) passes `(height, width)` as `orig_target_sizes`. On the downloaded ONNX graph, this placed boxes outside non-square frames. Passing `(width, height)` aligned its boxes with D-Fire annotations; for example, the flames in `AoF07935.jpg`. Only box coordinates changed: class scores and image/video hit counts did not. [detector.py](detector.py) contains the corrected mapping, and the overlap numbers above were computed after that correction.

## Video exploration

I downloaded six clips in the **published 70-clip video test list**: `FP1`, `FP14`, `FP31`, `VP1`, `VP3`, and `VP5`. The source provides clip names but no frame-level smoke/flame boxes or onset times. I sampled approximately one frame per second and ran all five models on identical frames. The table shows the **first candidate alert time in seconds from the clip start**, requiring two consecutive sampled frames with any smoke or flame prediction at score 0.25. A dash means no two-hit alert. It is **not** time from first visible smoke and does not establish true-positive video accuracy.

| Model | FP1 | FP14 | FP31 | VP1 | VP3 | VP5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-FINE M | — | — | 3 s | 2 s | 1 s | 1 s |
| D-Fire YOLOv8n | — | — | 5 s | 13 s | 1 s | 21 s |
| FireViewer YOLO11-M | — | — | 3 s | 21 s | 9 s | 1 s |
| SoulPerforms | — | 1 s | 3 s | — | — | — |
| PyroNear | — | — | — | — | 2 s | — |
| Smoke OR flame specialists | — | 1 s | 3 s | — | 2 s | — |

Visual inspection of the three FP clips found ordinary night scenes and traffic, with no visible fire. `FP31` produced persistent flame-like detections on vehicle lights for **four of five** models, despite the two-hit gate. `FP14` is only about 1.5 seconds long, so two sampled frames provide little temporal evidence. At score 0.50, D-FINE no longer has a two-hit `FP31` alert, but it also has no two-hit `VP3` alert. The VP clips contain distant, visually subtle candidates; some predicted boxes sit on trees or structures. Without verified onset and per-frame boxes, earlier model timestamps are **not evidence of earlier correct smoke detection**.

The [frame scores](outputs/dfire_benchmark/video_frames.csv) and [per-clip summary](outputs/dfire_benchmark/video_summary.csv) preserve every sampled timestamp. The [FP31 3-second D-FINE frame](outputs/dfire_benchmark/FP31_3s_dfine.jpg) illustrates a persistent night-light distractor. The local app lets you select these clips, inspect boxes frame by frame, vary score and persistence settings, and compare all five models.

I also replayed the saved 1 fps scores through class-specific one-hit, two-consecutive, three-consecutive and two-of-last-three rules at 0.25 and 0.50, with **no additional model inference**. At 0.25, the known night-light distractor in `FP31` still produces a three-consecutive candidate from D-FINE (4 s), D-Fire YOLOv8n (6 s), FireViewer YOLO11-M (4 s) and SoulPerforms (4 s). The same rule moves the `VP3` candidate to 2 s for D-FINE and D-Fire YOLOv8n, 13 s for FireViewer YOLO11-M, and 3 s for PyroNear Sensitive. This illustrates delay and persistence; `VP3` candidates have not been adjudicated as smoke. The rules require repeated hits of one class but do not yet require boxes to overlap the same region. The [full alert-rule sweep](outputs/dfire_benchmark/alert_rule_sweep.csv) and app contain all clips and cutoffs.

### Released PyroNear temporal smoke system

I also ran [PyroNear's v0.4.0 temporal package](https://huggingface.co/pyronear/temporal-model) on the same six clips at about 1 fps. Unlike the smoke-only still-image checkpoint above, this **complete system** bundles a different YOLO companion detector (`yolo11s_nimble-narwhal_v6.0.0`), tracks smoke candidate boxes across frames, and sends region crops through a DINOv2 image encoder plus transformer temporal head. Windows held at most 20 frames, with 10-second overlap. The package's calibrated decision threshold is 0.346.

All **10 windows were below its smoke-decision threshold**. It proposed no candidate boxes on any FP clip, including the misleading night traffic in `FP31`. It did propose and classify tubes on some VP windows: the strongest was `VP3` 0–19 s at probability **0.202**; `VP1` peaked at **0.100** and `VP5` at **0.010**. The zero decisions cannot be called true or false negatives without verified frame-level smoke truth. The bundled detector took about 5–7 s per 30-frame clip on this CPU; windows with candidate tubes added roughly 1–2.3 s for crop/temporal processing. The [window-level CSV](outputs/dfire_benchmark/pyronear_temporal_windows.csv) records all 10 verdicts and stage timings. This is **not** an ablation of the temporal classifier against the PyroNear Sensitive image detector: the companion YOLO weights differ. See [TEMPORAL_EXPERIMENT.md](TEMPORAL_EXPERIMENT.md) for the architecture comparison and next experiment sequence.

## What this says about the setups

- **Small D-Fire YOLOv8n:** fastest useful two-class baseline here, with few sampled clear-image hits. Its D-Fire training domain makes this a starting point for transfer testing, not a deployment score.
- **D-FINE M:** strongest image sensitivity and localization at 0.25, with many clear-image hits and about seven times the D-Fire CPU time. Raising its threshold reduces nuisance alerts and misses some detections.
- **Larger FireViewer YOLO11-M:** supports both classes but was slower and less sensitive on this D-Fire sample. Its score behavior differs from D-FINE despite both using the FireViewer corpus family.
- **Fire-only YOLO:** cheap and sometimes useful for visible flame, but structurally cannot surface smoke before flames appear. The Space title should not be read as proof that its actual weights detect smoke.
- **Wildfire smoke specialist:** PyroNear has a distinct target domain and smoke-only output. This sample shows weak overlap with D-Fire annotations; inspect its boxes on actual distant yard footage before deciding whether that specialization helps EMR.
- **Temporal layer:** repeated-frame confirmation eliminates some isolated mistakes but not persistent reflections, headlights or static smoke-like structures. The FP31 result is the practical example.
- **Hosted model:** a browser/API test is easy for public samples, but local batch comparability, rights, cost, rate limits and footage sharing need separate confirmation.

## Reproduce or extend

From the project folder, with [benchmark dependencies](requirements-benchmark.txt), Chromium (`python -m playwright install chromium`), and model weights downloaded as in [README.md](README.md):

```powershell
python fetch_dfire_splits.py
python fetch_dfire_archive.py
python fetch_dfire_videos.py
python benchmark_dfire_images.py
python benchmark_dfire_videos.py
python benchmark_dfire_localization.py
python summarize_dfire.py
python summarize_specialist_pair.py
python sweep_alert_rules.py
streamlit run app.py
```

The optional PyroNear temporal run uses its isolated Python 3.12/Torch environment. Setup and pinned package details are in [TEMPORAL_EXPERIMENT.md](TEMPORAL_EXPERIMENT.md).

The downloaders store media under Git-ignored `dataset/`; results and manifests are in Git-ignored `outputs/dfire_benchmark/`. Scripts resume completed image/video rows. To evaluate a different sample or updated checkpoint, use a new output directory or remove only that benchmark's result CSVs before rerunning; old rows otherwise remain by design.

The next meaningful EMR comparison requires authorized incident footage and several ordinary overnight camera-hours. Mark first **visibly actionable smoke**, human escalation time, and adjudicated false alerts, then compare missed incidents, lead time and false alerts per camera-hour on the same cameras. This D-Fire exercise establishes the candidate models and their failure modes; it cannot estimate those EMR outcomes.

The [cross-dataset report](CROSS_DATASET_REPORT.md) now tests the same local checkpoints on public FURG fire videos and CMU industrial-smoke clips, including PyroNear's released temporal pipeline on the CMU sample. Those small outside-D-Fire checks expose a much harder transfer problem than the D-Fire image scores alone suggest.
