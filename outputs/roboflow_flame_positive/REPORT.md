# Flame detection on positive Roboflow images

Source: [Fire Data Annotations v5](https://universe.roboflow.com/fire-detection/fire-data-annotations/dataset/5). The export has 663 positive test images. Exclude 226 with a source ID also in train, then take one image per remaining source ID: **413 images**.

All four flame-capable local checkpoints processed the same images. A *hit* means at least one flame detection at the specified output-score cutoff. *Aligned* means at least one flame detection overlaps a published flame box at the specified IoU. These are image-level positive recall measures, not precision, specificity, COCO mAP, or an incident-level estimate.

| Model | Hit at 0.25 | Aligned IoU≥0.30 at 0.25 | Aligned IoU≥0.50 at 0.25 | Hit at 0.50 | Aligned IoU≥0.50 at 0.50 | Median CPU ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D-Fire YOLOv8n | 376/413 | 337/413 | 281/413 | 300/413 | 232/413 | 61 |
| FireViewer D-FINE M | 397/413 | 381/413 | 347/413 | 378/413 | 292/413 | 457 |
| FireViewer YOLO11-M | 368/413 | 329/413 | 274/413 | 303/413 | 220/413 | 728 |
| SoulPerforms YOLOv8n (fire only) | 360/413 | 258/413 | 182/413 | 301/413 | 146/413 | 60 |

The largest annotated flame box covers at most 1.12% of the image in the smallest quartile and at least 16.05% in the largest quartile. Size-stratified counts are in `summary.csv`.

The images are all flame-positive. The exported labels have been format-checked and a small sample was visually reviewed, but the entire cohort has not been hand-corrected. The source-ID rule catches explicit Mirror/Noise copies; it does not prove separate incidents or exclude every near duplicate or model-training overlap. Output scores are not calibrated across models. The run cannot measure false alarms or alert timing; use verified negatives and whole clips for those.

`per_image.csv` preserves each model-image result, output score, box alignment and CPU latency. `cohort_images.txt` lists the exact evaluation images. Reproduce with `python benchmark_roboflow_flame_positive.py` and `python summarize_roboflow_flame_positive.py`.
