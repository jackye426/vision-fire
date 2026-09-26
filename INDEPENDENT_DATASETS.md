# Testing beyond D-Fire

The D-Fire YOLOv8n checkpoint was trained on the D-Fire image family. Its good D-Fire test-image result is useful for checking the local pipeline, but does not measure transfer to other cameras. The [first cross-dataset report](CROSS_DATASET_REPORT.md) now compares the local models on five FURG videos and six CMU industrial-smoke clips. A larger trial should use original media, keep whole videos/cameras together, check for duplicate or near-duplicate images against known training data where feasible, and report smoke and flame separately. Different repository names alone do not prove there is no source overlap.

| Public source | Labels and domain | Best use here | Access and caveat |
| --- | --- | --- | --- |
| [FURG fire-video dataset](https://github.com/steffensbola/furg-fire-dataset) | Videos with per-frame **flame rectangles**, plus fire-free clips; moving camera and visible flames | External video demonstration and flame localization test | Direct public GitHub files, CC0-1.0. It does not label smoke. Five sampled clips and XML files are under `dataset/furg/`. |
| [MIVIA LFDN](https://mivia.unisa.it/large-fire-dataset-with-negative-samples-lfdn/) | 36,554 images with flame/smoke boxes, including 4,154 negative images and deliberate headlights, reflections, dust and clouds | Best next **image** transfer and nuisance-alert test for both classes | Its site routes the full download through MIVIA's dataset-access page; exact training-source overlap still needs checking. |
| [CMU CREATE Lab Deep Smoke Machine](https://github.com/CMU-CREATE-Lab/deep-smoke-machine) | Public labeled industrial-emissions video clips; positive/negative **smoke at clip level** | More relevant smoke domain than wildfire imagery; test smoke alert episodes over video | Six sampled videos and metadata are under `dataset/cmu/`. Clip labels do not provide pixel boxes or first-smoke timing. |
| [AI For Mankind / HPWREN smoke set](https://github.com/aiformankind/wildfire-smoke-dataset) | Distant-camera wildfire smoke images with Pascal VOC boxes; version 2 has 2,192 annotated images | Smoke-only distance and faint-plume stress test | Wildfire domain differs from EMR; attribution and noncommercial share-alike terms apply. Check overlap with any PyroNear training data before treating it as independent for that model. |
| [Roboflow Fire Detection workspace](https://universe.roboflow.com/fire-detection) | Mostly flame-only image projects, with some empty projects | Small **exploratory flame-image** challenge | The downloaded v5 export maps to one `Fire` class and contains only positive images. See the [local ZIP audit](ROBOFLOW_DATASET_AUDIT.md). |

The [MIVIA Fire Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) is another video option: 31 clips, 14 with fire and 17 without, including confusing negatives. It has video-level labels rather than the FURG frame rectangles. These collections serve different questions; a single leaderboard across them would obscure domain and label differences.

## Roboflow Fire Detection workspace audit (25 September 2026)

The linked [workspace](https://universe.roboflow.com/fire-detection) lists seven projects, but three have **zero images** (`tsd`, `safety`, `fire smoke`) and `ok` has only **15**. None of the populated projects supplies a usable smoke class or linked video incidents, so they cannot test early-smoke or temporal alert performance.

| Project | What is actually available | Decision |
| --- | --- | --- |
| [Fire Data Annotations, v5 `original_raw-images`](https://universe.roboflow.com/fire-detection/fire-data-annotations/dataset/5) | 3,284 images; 2,621 train, 663 test, no validation or augmentation. The **downloaded export** has one `Fire` class, valid YOLO box lines, and no empty labels. | **Candidate for a curated flame-positive trial.** The source browser lists historical `fire` and `0` classes, but `data.yaml` in v5 resolves them to `Fire`. The [unannotated source image](https://universe.roboflow.com/fire-detection/fire-data-annotations/images/00Rgxheu5ivv391EurPV) is not evidence of a missing label in v5: every exported image has a box. No negative images are present, so specificity needs another set. See the [ZIP audit](ROBOFLOW_DATASET_AUDIT.md). |
| [fire detection data pre, v4](https://universe.roboflow.com/fire-detection/fire-detection-data-pre/dataset/4) | 2,000 resized 416×416 images; 1,402 train, 401 validation, 197 test; only `fire` class. A hosted model is attached to the same project. | Secondary flame-only transfer sample. Do not use its own test set to claim independent performance for its attached model. |
| [fire, v1](https://universe.roboflow.com/fire-detection/fire-bxkxw/dataset/1) | Project lists 2,055 source images, but the downloadable v1 has **501** images: 438 train, no validation, 63 test. Training outputs include rotation/brightness augmentation. Only `fire` class. | Low priority: small test split and apparent overlap with `fire detection data pre` (both project previews show the same image IDs). |

The [downloaded Roboflow audit](ROBOFLOW_DATASET_AUDIT.md) now covers four exports. For this workspace, use **v5 raw** for flame-positive localization, inspect a larger stratified subset and supplement it with independently sourced negative images. Keep visually related images from the same source/scene together and check near-duplicates against model training sources. MIVIA LFDN remains more suitable for a combined smoke/flame and hard-negative benchmark.

## D-Fire YOLOv8n on an external fire clip

I rendered D-Fire YOLOv8n over FURG's `hand_held_camera_wildfire.mp4`, a **25.8-second vehicle-fire clip** with moving-camera footage. The [annotated H.264 video](outputs/furg/hand_held_camera_wildfire_dfire_yolov8n_h264.mp4) shows orange flame and green smoke boxes at model output score ≥ 0.25, sampled at about 10 fps. FURG's XML first marks a flame on the sampled frame at **1.301 s**; the model's first flame box appears at **2.002 s**. It emits a flame box on **224 of 245 sampled frames** where FURG marks flame. The model also emits smoke boxes, but FURG does not label smoke, so those cannot be scored. During a brief camera pan, the fire moves partly off screen and the model drops flame detections.

This is **one visual case study**, not a cross-dataset recall estimate. We have not audited whether D-Fire training images include near-duplicates of this older public video. The [per-frame results](outputs/furg/hand_held_camera_wildfire_dfire_yolov8n.csv) are saved locally. The Streamlit app embeds the video and lets you run any of the five detectors interactively on the original FURG clip.

To reproduce after installing `requirements-benchmark.txt` and the D-Fire YOLO model:

```powershell
python fetch_furg_examples.py
python render_furg_fire.py
python transcode_furg_fire.py
```

For a broader transfer check, sample MIVIA LFDN by smoke-only, flame-only, both and hard-negative categories; keep thresholds fixed from a separate validation sample. Expand the FURG and CMU video samples by independent incident and camera, and manually mark first actionable smoke and nuisance events where the published labels do not supply them.
