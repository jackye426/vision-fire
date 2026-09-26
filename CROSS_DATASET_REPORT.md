# First checks outside the D-Fire images

Run on 25 September 2026. These are **small, exploratory transfer checks**, not a claim about EMR camera performance or a statistically reliable ranking. The [D-Fire image benchmark](BENCHMARK_REPORT.md) favored a checkpoint trained on that dataset family; this report asks how the same lab models behave on two different public sources. Original media, model files and results are under Git-ignored `dataset/`, `models/` and `outputs/`.

## 1. [FURG fire videos](https://github.com/steffensbola/furg-fire-dataset): annotated flame

I sampled about **one frame per second** from five FURG clips: `barbecue`, `hand_held_camera_wildfire`, `house1`, `non_fire_patrolbot_onboard` and `coolerbot`. The first three have FURG flame rectangles; the last two have none. This gave **79 sampled frames with a flame annotation** and **102 sampled frames from two flame-free clips**. FURG labels flame only, so smoke boxes cannot be scored here. “Hit” means any flame box on a labeled frame; “aligned” requires an IoU of at least 0.50 with a FURG flame rectangle. All five local checkpoints used the same original frames, with a 0.25 model-output cutoff.

| Model | Flame frames hit | Flame boxes aligned | Flame flags on 102 frames in the two flame-free clips | Flame-free clips with a two-consecutive-hit alert | Median CPU ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| D-Fire YOLOv8n, smoke + flame | 62/79 | 51/79 | 4/102 | 1/2 | 54 |
| D-FINE M, smoke + flame | **71/79** | **57/79** | 3/102 | 0/2 | 375 |
| FireViewer YOLO11-M, smoke + flame | 65/79 | 55/79 | **0/102** | 0/2 | 697 |
| SoulPerforms, flame only | **71/79** | 55/79 | **15/102** | 2/2 | 55 |
| PyroNear Sensitive, smoke only | Not supported | Not supported | Not supported | Not supported | 319 |

At cutoff **0.50**, D-Fire YOLOv8n falls to 53/79 flame hits, D-FINE to 68/79, FireViewer YOLO11-M to 57/79, while SoulPerforms stays at **71/79**. None of the four flame-capable checkpoints flags a flame on the two flame-free clips at 0.50. This illustrates threshold sensitivity, not a calibrated equal-false-alarm comparison. For the visually small `house1` fire, FURG first marks flame at a sampled **5 s**; the first 0.25 flame box comes at **19 s** for D-Fire YOLOv8n, **13 s** for D-FINE/SoulPerforms and **16 s** for FireViewer YOLO11-M. On the large vehicle fire, all dual-class models show a flame box by the first 1 fps frame with a flame label (2 s). These frames are correlated within just **three incidents**.

The [annotated vehicle-fire video](outputs/furg/hand_held_camera_wildfire_dfire_yolov8n_h264.mp4) shows D-Fire YOLOv8n at about 10 fps. Its [per-frame scores](outputs/furg/frame_results.csv), [aggregate table](outputs/furg/summary.csv), and [clip summary](outputs/furg/clip_summary.csv) are saved. The 10 fps visual render has its first FURG flame label at 1.301 s and first model flame box at 2.002 s; its frame count differs from the 1 fps common comparison above.

## 2. [CMU CREATE Lab industrial smoke videos](https://github.com/CMU-CREATE-Lab/deep-smoke-machine): clip-level smoke

I selected one **strong-positive** and one **strong-negative** clip for each of three cameras from the project's public metadata, requiring both volunteer and researcher label states to agree, then taking the lowest metadata ID in each group. The sample has **three smoke-labelled and three clear-labelled clips**. Each downloaded 320×320 video has 36 time-lapse frames; consecutive source images represent roughly five real-world seconds. The [sample manifest](dataset/cmu/sample_manifest.csv) records every source URL, label and hash. CMU provides **clip-level smoke labels**, not smoke boxes or exact onset times. Some clips labelled clear visibly contain pale plumes or steam: this is a demanding distinction that may differ from an EMR operator's definition of actionable smoke.

For frame detectors, the table counts a clip when it has **two consecutive frames with a smoke box** at output-score cutoff 0.25. The PyroNear temporal row uses its **packaged** decision threshold and its own companion YOLO, with two 20-frame windows per clip. Its CPU time is not directly comparable to the ONNX frame detectors.

| Smoke-capable setup | Smoke-labelled clips flagged | Clear-labelled clips flagged | Smoke-box frames in 108 smoke-labelled frames | Smoke-box frames in 108 clear-labelled frames | Median CPU ms/frame* |
| --- | ---: | ---: | ---: | ---: | ---: |
| D-Fire YOLOv8n | 1/3 | 2/3 | 5/108 | 52/108 | 52 |
| D-FINE M | **3/3** | **3/3** | 108/108 | 84/108 | 382 |
| FireViewer YOLO11-M | 2/3 | 2/3 | 33/108 | 71/108 | 698 |
| PyroNear Sensitive | 1/3 | 2/3 | 7/108 | 14/108 | 325 |
| PyroNear temporal v0.4.0 | 1/3 | **0/3** | Window verdict only | Window verdict only | Different pipeline |

*Median time includes frame preprocessing, inference and postprocessing for the ONNX detectors, with model initialization excluded. The PyroNear temporal system's companion YOLO took roughly 10–16 s per 36-frame clip on this CPU, plus its window processing. SoulPerforms is a flame-only checkpoint and has **no smoke output**, so it is omitted from the smoke table. The simple smoke-specialist + flame-specialist OR combination does not change the smoke-class figures of PyroNear Sensitive.

At score **0.50**, the two-consecutive smoke results are: D-Fire YOLOv8n **0/3 positive, 1/3 clear**; D-FINE **3/3 positive, 2/3 clear**; FireViewer YOLO11-M **1/3 positive, 2/3 clear**; and PyroNear Sensitive **1/3 positive, 0/3 clear**. PyroNear temporal remains **1/3 positive, 0/3 clear** at its packaged threshold. The source clips are too few to claim that a temporal architecture beats threshold tuning. The [CMU frame results](outputs/cmu/frame_results.csv), [clip summary](outputs/cmu/clip_summary.csv), [aggregate](outputs/cmu/aggregate.csv), and [PyroNear temporal windows](outputs/cmu/pyronear_temporal_windows.csv) preserve the decisions. The [scene contact sheet](outputs/cmu_contact.jpg) and [one D-Fire false candidate on a clear-labelled plume](outputs/cmu/clear_cam1_26870_dfire_f18.jpg) help interpret the labels.

## What changed from the D-Fire-only picture

- D-Fire YOLOv8n remains the **fastest useful dual-class model**, but its D-Fire-image success did not carry over cleanly to small FURG flames or CMU industrial smoke. It misses the first 14 sampled seconds of the small `house1` flames and its smoke rule flags two of three CMU clear clips at 0.25.
- D-FINE is more flame-sensitive on FURG, yet at 0.25 it flags **every** CMU clear-labelled clip as smoke. The core tradeoff is sensitivity versus nuisance alerts, not a single overall percentage.
- SoulPerforms is a strong **flame specialist** on these FURG clips, especially at 0.50, but it has no smoke class and flagged both flame-free clips at 0.25. Combining it with a smoke specialist only adds the separate smoke detector's coverage and cost.
- PyroNear's full temporal release is conservative on the selected CMU clips: no clear clip verdicts, but two smoke-labelled clips missed. It uses a different companion detector from PyroNear Sensitive, so this is not a clean temporal-head ablation.

We have **not** yet run the complete pedbrgs AVT/TPT system, Roboflow's hosted model, a vision-language model, SlowFast, or Video Swin on these sources. We also have not audited near-duplicates between D-Fire training images and the older FURG/CMU media. The next credible step is a larger, preselected set of independent events and long clear camera periods, with human labels for first actionable smoke, flames, steam/dust and hard negatives. The MIVIA LFDN image set is a promising larger flame/smoke transfer check, but its download currently routes through MIVIA's dataset-access page.

## Reproduce

With the existing model downloads and benchmark requirements installed:

```powershell
python fetch_furg_examples.py
python benchmark_furg_videos.py
python summarize_furg.py
python render_furg_fire.py
python transcode_furg_fire.py
python fetch_cmu_metadata.py
python fetch_cmu_smoke_sample.py
python benchmark_cmu_smoke.py
python summarize_cmu_smoke.py
```

The metadata downloader verifies the project's [published metadata snapshot](https://github.com/CMU-CREATE-Lab/deep-smoke-machine/blob/master/back-end/data/dataset/2020-02-24/metadata_02242020.json) by SHA-256. The optional temporal check uses the isolated Python 3.12 environment from [TEMPORAL_EXPERIMENT.md](TEMPORAL_EXPERIMENT.md):

```powershell
.\.venv-temporal\Scripts\python.exe benchmark_pyronear_temporal_cmu.py
```
