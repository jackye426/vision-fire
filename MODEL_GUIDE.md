# Existing smoke and fire vision models: what differs

The [EMR Tilbury visit notes](https://app.notion.com/p/3e5196f5637381aea9c8e09aeb82013a) suggest a specific test: how early can existing CCTV surface visible smoke to the human monitoring team without flooding them with false alerts? The East Tilbury overnight fire footage is the useful retrospective case if EMR authorises access. A model's published image score cannot answer this on its own.

**Model types:** A frame **classifier** says whether fire appears somewhere in an image; the pedbrgs repository includes FireNet and MobileNet classifiers as baselines. An object **detector** draws boxes and class scores; YOLO and D-FINE models here are detectors. A **temporal system** adds video behavior to a detector. Roboflow's YOLO26 is another detector accessed through a hosted product, not a separate sensing modality. None of these models reads physical temperature from a thermal sensor.

## Four setup choices

1. **Visible target.** A fire-only detector can be useful for obvious flame, but cannot find the earlier smoke window. A smoke-only detector may find that window but cannot confirm flame. Two-class detectors attempt both.
2. **One frame or a sequence.** A frame detector reacts quickly but can fire on a single confusing image. Requiring repeated detections can reject transient noise but delays the alert. More advanced temporal methods check whether a candidate region changes like smoke or fire.
3. **Training domain.** Wildfire cameras, domestic scenes, web images and industrial CCTV have different scale, lighting, dust, exhaust, steam, machinery and night conditions. Transfer to scrap yards must be measured.
4. **Where inference runs.** Local ONNX keeps footage on this machine. A hosted demo or API is easy to try but sends the uploaded material to another service. Open code, downloadable weights and commercial rights are separate questions.

## The models and systems in this trial

| Source | Type and setup | Strength to test | Main limitation | Evidence available |
| --- | --- | --- | --- | --- |
| [D-Fire YOLOv8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) | Small YOLOv8n, smoke + fire, local PyTorch checkpoint exported here to 640px ONNX | Fast two-class baseline and simple local deployment | D-Fire domain; no temporal reasoning; AGPL-3.0 | Publisher reports mAP@50 **0.754**, precision **0.766**, recall **0.688** on its own 4,306-image test split |
| [SoulPerforms Gradio demo](https://huggingface.co/spaces/SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio) | Hosted browser demo; its 640px YOLOv8n ONNX file also runs locally here | Immediate visual test and fire-only baseline | The actual ONNX label map contains only **fire**, despite the Space name; no benchmark protocol reported; ONNX metadata says AGPL-3.0 while Space config says MIT | Public code and weights, no documented independent performance figure |
| [PyroNear Sensitive Detector](https://huggingface.co/pyronear/yolo11s_sensitive-detector) | 1024px YOLO11s, smoke only, local ONNX | Early, distant plume specialist; Apache-2.0 model release | Wildfire domain may not transfer to yard smoke; no flame class; more pixels cost CPU time | Model card provides training and version provenance, but no comparable EMR-style score |
| [pedbrgs research system](https://github.com/pedbrgs/Fire-Detection) | YOLOv5 spatial detection followed by area variation (AVT) or temporal persistence (TPT), run through Docker | Tests whether motion and persistence suppress transient false alarms | Extra delay and configuration; original weights are separate downloads; this lab's simple streak control is **not** the paper's implementation | Research code and paper; no direct common evaluation with the models above |
| [Roboflow community model](https://universe.roboflow.com/s-workspace-173hq/fire-smoke-detection-avnev-goqj2) | Hosted YOLO26 fire + smoke model, browser/API with key | Very easy to inspect visually; useful external comparator | Weight export and commercial rights unconfirmed; hosted upload is external; own-dataset metric not comparable | Page reports mAP@50 **0.738**, precision **0.822**, recall **0.657** on its 4,226-image dataset |
| [FireViewer D-FINE M](https://huggingface.co/fireviewer) | 704px two-class detector, local ONNX | A second architecture with an independent image evaluation | Larger than YOLOv8n; published results still do not cover EMR cameras | Publisher reports independent mAP@50–95 **0.3924**, calibrated holdout F1 **0.6475** |
| [FireViewer YOLO11-M](https://huggingface.co/fireviewer/fire-smoke-yolo11m-strict-v1) | 960px two-class detector, local ONNX | Architecture comparison against D-FINE on the same FireViewer protocol | Its own card reports weak flame AP and substantial negative-image false alarms; AGPL-3.0 | Independent mAP@50–95 **0.1491** and negative-image false-alarm rate **0.3391** on the published protocol |

The D-Fire, Roboflow and FireViewer numbers use different test sets and sometimes different metrics; **do not rank the systems by those published percentages**. The FireViewer D-FINE and YOLO11-M figures are from the same published independent image protocol, but even that protocol is not an EMR night-camera test.

The [D-Fire benchmark report](BENCHMARK_REPORT.md) now supplies a common local trial: five downloadable checkpoints on the same 160 labeled D-Fire test images and six published test clips. It includes image recall, labeled-box overlap, clear-image flags, CPU time, and video candidate-alert timing. D-Fire-trained YOLOv8n has a domain advantage on this corpus; the video split lacks frame-level annotations, so those timings need visual review.

The [temporal experiment plan](TEMPORAL_EXPERIMENT.md) extends this comparison to a smoke-specialist/fire-specialist OR rule and the released [PyroNear temporal smoke system](https://huggingface.co/pyronear/temporal-model). That system uses its own YOLO detector, links boxes into tubes, and classifies their evolution with a DINOv2 encoder and transformer; it was run on ten windows from the six D-Fire clips. SlowFast, Video Swin and a vision-language model are described as later experiments, with **no measured fire/smoke score yet**. Their architecture alone is not evidence that they will outperform a well-tuned frame detector.

The [cross-dataset report](CROSS_DATASET_REPORT.md) now shows why the D-Fire image result is not enough: on five external FURG videos, the fastest dual-class D-Fire YOLOv8n finds 62/79 sampled annotated flame frames at score 0.25; on six CMU industrial-smoke clips, its two-hit smoke rule flags 1/3 smoke-labelled clips and 2/3 clear-labelled clips. These are small, correlated samples with different label types. D-FINE is more sensitive, while PyroNear's packaged temporal system is more selective on the CMU sample, and each has misses or nuisance alerts that still need an EMR-relevant test.

## A useful first comparison

Use the local app to run the same footage through the available checkpoints with fixed sampling and record:

| Positive clips | Clear clips | Practical cost |
| --- | --- | --- |
| First visible smoke; first raw smoke hit; first persistent alert; first flame; first human escalation; missed events | False alert episodes and flagged sampled frames on ordinary nights, dust, steam, exhaust, hot work, lights and sun glare | CPU milliseconds per sampled frame, sampling interval, and persistence delay |

Start with a few independent positive clips and several clear clips to learn failure modes. Treat a small trial as qualitative. A credible EMR pilot needs enough ordinary camera-hours and separate incidents to estimate false alerts per camera-hour and missed events. Keep camera, time of day, visibility and weather attached to each result. RGB image detectors cannot read real temperature from false-colour thermal feeds or see heat hidden inside a pile.

The current lab includes **a simple consecutive-frame alert**, which is a way to explore the speed/false-alarm tradeoff. It does not reproduce the pedbrgs AVT or TPT algorithms. Its first persistent alert is timestamped at the confirming frame, so the delay is visible in the table. For a known clear clip, tick **This entire clip is known to be clear** to count false alert episodes.

The included D-Fire collage has boxes drawn into the image. It is only a pipeline check; it is not a fair benchmark image. Do not send EMR footage to hosted demos or APIs without EMR's footage-sharing agreement.
