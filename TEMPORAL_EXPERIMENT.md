# Fire and smoke video approaches: experiment plan

The [EMR visit notes](https://app.notion.com/p/3e5196f5637381aea9c8e09aeb82013a) make **early visible smoke with a manageable false-alert load** the primary question. Flame detection remains a useful second signal. This plan uses the [D-Fire repository](https://github.com/gaia-solutions-on-demand/DFireDataset) for public exploration, then reserves any claim about EMR performance for representative, authorised camera footage.

## Available D-Fire material

The repository links the original labeled image archive, published train/validation/test lists, a [Kaggle ready-to-use copy](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo), surveillance video, and [research code with trained models](https://github.com/pedbrgs/Fire-Detection). It directs researchers to request more surveillance access through [Apaga o Fogo](https://apagaofogo.eco.br/). The local lab already has the original image archive, split lists, and six published test videos. The public video split does **not** supply frame boxes or verified first-smoke timestamps; clip names alone cannot establish detection accuracy or lead time.

## Architectures worth comparing

| Approach | What it actually consumes and outputs | Why try it | Main cost or blind spot | Current lab status |
| --- | --- | --- | --- | --- |
| Single-frame dual-class detector, such as [D-Fire YOLOv8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) or [D-FINE](https://huggingface.co/fireviewer/fire-smoke-dfine-m-strict-v1) | Each RGB frame to smoke/flame boxes | Simple, fast baseline and visible localization | No motion evidence; lights and steam can look convincing | Measured on 160 images and six clips |
| Repeated-frame YOLO with an alert rule | Same boxes, then consecutive hits or a *k-of-n* window, ideally for one class and tracked region | Near-zero training cost; can tune delay against nuisance alerts | Persistent headlights/reflections still pass; sampling can miss short events | Two consecutive sampled hits measured; app can vary threshold and streak |
| Two specialists together | Smoke-only detector OR flame-only detector; optional per-class thresholds | Tests whether a distant-smoke model adds value without sacrificing flame | Two inference passes; the OR rule can add false positives | PyroNear Sensitive + SoulPerforms decision-level OR measured |
| Detector plus temporal verification, e.g. [pedbrgs](https://github.com/pedbrgs/Fire-Detection) | YOLO candidates followed by area variation or temporal persistence on video | Uses motion and shape change to reject some static distractors | Depends on candidate recall and chosen time window | Published code/weights identified; full AVT/TPT pipeline not yet run |
| Image encoder plus temporal head, e.g. [PyroNear temporal v0.4.0](https://huggingface.co/pyronear/temporal-model) | Its companion YOLO links smoke boxes into tubes; 224px crops pass through DINOv2 ViT and a two-layer transformer, then a logistic decision head | Learns how smoke candidates evolve while spending the costly encoder on proposed regions | Smoke only; wildfire domain; detector misses cannot be repaired by the head | Full published package run on six clips, below |
| Full-frame video backbone, e.g. [SlowFast](https://github.com/facebookresearch/slowfast) or [Video Swin](https://github.com/SwinTransformer/Video-Swin-Transformer) | Video clip to a trained class score; SlowFast uses slow spatial and fast motion pathways, Video Swin uses local space-time attention | Could learn wider scene/motion context beyond detector boxes | Public checkpoints are mainly action-recognition weights; fire/smoke needs task-specific labels and fine-tuning; more compute | Architecture candidates, **no fire/smoke performance measured** |
| Vision-language model (VLM), e.g. [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Image or sampled video frames plus a prompt to smoke/flame/uncertain labels and explanation | Flexible diagnostic baseline; can inspect ambiguous cases without training | Slower/costlier, prompt-sensitive, weak spatial localization and calibration; hosted use transmits footage | Proposed only; **no local VLM performance measured** |

A text-only LLM cannot inspect pixels. A VLM can classify frames or clips, but a free-form answer is not a calibrated detector score. Video VLM input must specify sampling and frame budget, and all results need the same clip-level label rules as the other systems.

## New measured comparisons

### Specialist smoke OR specialist flame

I combined the existing [PyroNear Sensitive smoke-only detector](https://huggingface.co/pyronear/yolo11s_sensitive-detector) and the [SoulPerforms flame-only checkpoint](https://huggingface.co/spaces/SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio) without retraining. Each class keeps its own model; either class can flag the image. On the same balanced 160 D-Fire test images at the shared **0.25 output-score cutoff**:

| Setup | Smoke hit | Flame hit | Clear images flagged | Same-class box aligned at IoU ≥ 0.50 | Median summed CPU time |
| --- | ---: | ---: | ---: | --- | ---: |
| Smoke + flame specialists, OR | 19/80 | 52/80 | 6/40 | Smoke 8/80; flame 36/80 | 373 ms/image |
| D-Fire YOLOv8n, dual-class | 71/80 | 74/80 | 0/40 | Smoke 64/80; flame 67/80 | 53 ms/image |
| D-FINE M, dual-class | 76/80 | 79/80 | 25/40 | Smoke 68/80; flame 73/80 | 385 ms/image |

At cutoff 0.50, the specialist pair hits 10/80 smoke and 39/80 flame images, with 3/40 clear flags. On the sampled videos at 0.25, its first two-hit candidate was `FP14` 1 s, `FP31` 3 s and `VP3` 2 s; there was none for `FP1`, `VP1` or `VP5`. These are candidate decisions on clips without frame-level event truth. The pairing is **not** evidence that specialist models are generally worse: these particular weights have different training domains, and the D-Fire YOLOv8n model has a direct dataset-family advantage. The pair illustrates the latency and false-alert cost of a naive OR rule.

### PyroNear temporal smoke package

I ran the [released v0.4.0 package](https://huggingface.co/pyronear/temporal-model) at about **1 frame/second** on all six clips, in windows of up to 20 frames (10-second overlap for clips longer than 20 seconds). This package uses **`yolo11s_nimble-narwhal_v6.0.0`**, not the PyroNear Sensitive checkpoint in the still-image trial. Its config proposes boxes at YOLO score 0.1 and 1024px, tracks candidate tubes, and makes a calibrated smoke decision at **0.346**. This is an end-to-end released system test, not an isolation of the temporal head.

| Clip | Windows | Companion YOLO frames with candidates | Strongest kept-tube probability | Smoke-positive windows |
| --- | ---: | --- | ---: | ---: |
| `FP1` | 2 | 0 in either | 0 | 0 |
| `FP14` | 1 | 0 | 0 | 0 |
| `FP31` | 1 | 0 | 0 | 0 |
| `VP1` | 2 | 0, then 6/20 | 0.100 | 0 |
| `VP3` | 2 | 12/20, then 2/20 | **0.202** | 0 |
| `VP5` | 2 | 0, then 2/20 | 0.010 | 0 |

Thus the complete system **rejected all ten sampled windows**, including the persistent `FP31` night-light distractor, but also did not emit an alarm on the VP windows. Some VP windows generated and classified tubes, so the zero verdict is not simply a failed runtime. Without adjudicated smoke onset, those VP negatives cannot be scored as correct misses or correct rejections. The companion detector took about **5–7 seconds per 30-frame clip** on this CPU; a window with kept tubes added roughly **1–2.3 seconds** for crops and temporal classification. Different input resolution and PyTorch execution make these timings unsuitable as a pure architecture speed comparison with ONNX detectors. Raw data: [pyronear_temporal_windows.csv](outputs/dfire_benchmark/pyronear_temporal_windows.csv).

### Cheap temporal-rule replay

Using the saved frame scores, I tried one hit, two consecutive hits, three consecutive hits, and two of the last three sampled frames, with the **same class** required across hits. This needs no additional neural inference. On the visually clear night-traffic clip `FP31`, three consecutive hits still produced candidates from D-FINE, D-Fire YOLOv8n, FireViewer YOLO11-M and SoulPerforms. Repeated-frame counting alone cannot reject a persistent headlight. On `VP3`, three hits delayed the first candidate by 1–7 seconds versus two hits, depending on model; whether those candidates are actual smoke is unverified. The [full rule sweep](outputs/dfire_benchmark/alert_rule_sweep.csv) includes both tested score cutoffs. A next rule should require overlapping boxes from the same tracked region.

## Fastest fair experiment sequence

1. **Create event-level truth.** Use D-Fire training/validation material for development, and annotate a small, diverse set of videos at the first *visibly actionable smoke*, visible flame, and no-event periods. Include day/night, distant plumes, headlights, steam, dust, exhaust and hot work. Keep whole events and cameras together when splitting data. The six test clips already explored here are discovery examples; do not tune against them and later call them an untouched test set.
2. **Replay the existing detector scores.** Compare one hit, consecutive hits, and class-specific *k-of-n* rules without rerunning the neural nets. Sweep each model's threshold on validation clips, then freeze it. Add region tracking or box overlap so smoke in one place and a later headlight elsewhere cannot satisfy the same persistence rule. Report an alert when the confirming frame arrives, including the sampling delay.
3. **Check specialist fusion and low-cost temporal verification.** Try separate smoke/flame thresholds, an OR rule, and a rule that asks the temporal model to verify only smoke candidates. Run the pedbrgs AVT/TPT implementation if its released pipeline and weights are reproducible. Compare it with the simple gate on identical frames, candidate proposals, and event truth. Retain the published PyroNear package as a complete-system comparator; replacing its companion detector would create a separate, out-of-distribution experiment.
4. **Use a small VLM diagnostic set.** Once a video-capable VLM is available, use a fixed prompt and frame budget on the same public positive and hard-negative windows. Request structured `smoke`, `flame`, `uncertain` labels plus short evidence, and record latency/cost. Compare image-only prompts with multi-frame prompts. Treat boxes and first-smoke timing as separate questions. Keep EMR footage on an authorised processing path.
5. **Fine-tune full-video backbones only if simpler stages stall.** SlowFast and Video Swin checkpoints are useful starting weights, but their public action classes are not fire/smoke outputs. They require enough independently labeled video incidents, training compute, and a same-data comparison against a small temporal head. A few clips are enough to learn failure modes, not to claim superiority for a trained video backbone.

For each system, keep **smoke recall, flame recall, missed incidents, time from first actionable visible smoke to alert, false alert episodes per camera-hour, CPU/GPU cost, and memory** separate. Also report score thresholds and frame sampling. Image-level clear flags are useful for screening; they are not a substitute for false alerts over long CCTV runs. The comparison that would matter at EMR is an independent replay of authorised historical incidents plus ordinary overnight camera-hours.

## Reproduce the new runs

The specialist OR calculation reuses the existing per-image, localization, and per-frame CSVs:

```powershell
python summarize_specialist_pair.py
python sweep_alert_rules.py
```

PyroNear's temporal core requires Python 3.11 or 3.12. In this workspace, Python 3.12 lives under Git-ignored `.uv-python/` and the isolated `.venv-temporal/` has `temporal-model-core[torch]` installed from PyroNear Git commit `fb7f6ca855cafa8a2ca0981ac54487c63892f6ad`. If the environment needs to be recreated, install [uv](https://docs.astral.sh/uv/) and run in PowerShell from this folder:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.uv-python'
uv python install 3.12
uv venv --python 3.12 .venv-temporal
uv pip install --python .\.venv-temporal\Scripts\python.exe 'temporal-model-core[torch] @ git+https://github.com/pyronear/temporal-model.git@fb7f6ca855cafa8a2ca0981ac54487c63892f6ad#subdirectory=core'
```

Then run:

```powershell
python fetch_pyronear_temporal.py
.\.venv-temporal\Scripts\python.exe benchmark_pyronear_temporal.py
```

The downloader pins Hugging Face revision `8b97c9f497c8cdfbda38b2bab21a00485ee8671e` and verifies SHA-256 `0fb060ca81a6e36944f28f80e81b1ae9e5b8557b207aca8a600fb9635547cb28`. The model ZIP, extracted frames and benchmark CSV are Git-ignored. The [PyroNear model card](https://huggingface.co/pyronear/temporal-model) describes an ONNX temporal export, but this run used the bundled Torch package so its companion detector and classifier followed the published end-to-end path.
