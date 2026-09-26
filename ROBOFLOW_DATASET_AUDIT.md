# Roboflow fire and smoke exports: local usability audit

25 September 2026. These are the actual YOLOv8 ZIP exports, not only Universe previews. Each ZIP includes images, labels, `data.yaml`, a license declaration and train/test directories. Keep the ZIPs as the source copies; `audit_roboflow.py` reads them directly and writes `audit.json` plus a labeled contact sheet beside each archive.

| Export | Local archive | Test labels | Verdict for this lab |
| --- | --- | --- | --- |
| [Fire Project, Fire and Smoke v8](https://universe.roboflow.com/fire-project/fire-and-smoke-lkzok/dataset/8) | `dataset/roboflow/fire-project-fire-and-smoke-v8-yolov8.zip` | 1,149 images; 688 with `fire`, 481 with `smoke`, 192 with `other`, 47 with empty labels | **Exploratory fire/smoke image challenge after curation.** It has useful scene variety, but at least one apparent fire is unlabeled and video frames cross its train/test split. |
| [MTSW Fire & Smoke Bounding Box v5](https://universe.roboflow.com/mtsws-workspace/fire-smoke-bounding-box-v8tif/dataset/5) | `dataset/roboflow/mtsw-fire-smoke-v5-yolov8.zip` | 1,615 images; 545 with fire, 1,026 with smoke, 482 empty | **Exclude from independent testing.** It is overwhelmingly D-Fire imagery, including all its validation and test images. |
| [AI For Mankind mirror v1](https://universe.roboflow.com/s-workspace-gfuym/fire-smoke-detection-gecep/dataset/1) | `dataset/roboflow/aiformankind-fire-smoke-v1-yolov8.zip` | 1,146 images; four case-variant classes; only one empty label | **Curated positive smoke/fire examples only.** Normalize classes, convert mixed polygon labels, remove duplicates and fix missing labels. It cannot measure false-alarm rate. |
| [Fire Data Annotations v5 raw](https://universe.roboflow.com/fire-detection/fire-data-annotations/dataset/5) | `dataset/roboflow/fire-data-annotations-v5-yolov8.zip` | 663 images, every one with `Fire` boxes; no smoke or negative images | **Usable for an exploratory flame-only recall/localization check.** Add an independent negative set for specificity. |

## What the files show

### Fire Project: 11,931 images

The YOLO export declares classes `fire`, `other`, `smoke` and contains exactly 8,468 train, 2,314 validation and 1,149 test image/label pairs. Label lines parse as valid YOLO boxes. The [contact sheet](dataset/roboflow/fire-project-fire-and-smoke-v8-yolov8/contact_sheet.jpg) shows plausible flame and smoke boxes and `other` boxes on confusing objects. It also shows a test image named `middle_-3208-...` with obvious flames and **no boxes**. Thus at least some of the 47 empty test labels are false negatives in the ground truth; counting model detections on those as false alarms would be wrong.

File naming reveals frames from the same apparent video in different splits: 78 inferred clip names span two or more splits, and 343 of 1,149 test images belong to those names. This is an inference from filenames, not a verified incident ID. At least 215 normalized filenames are also perceptually near-identical to images in the AI For Mankind mirror. To use this set for a reliable comparison, manually review and lock a subset, remove known cross-source copies, group video frames by incident, and score `other` separately rather than treating it as smoke or fire.

### MTSW: 16,352 images

The export has 13,110 train, 1,627 validation and 1,615 test image/label pairs, with `fire` and `smoke` classes. The [contact sheet](dataset/roboflow/mtsw-fire-smoke-v5-yolov8/contact_sheet.jpg) shows plausible boxes and clear negatives. However, **16,324 of 16,352** normalized filenames match our D-Fire ZIP. Every MTSW validation and test filename matches a D-Fire **test** filename; the other 13,082 matched names are in D-Fire train, leaving only 28 unmatched MTSW train names. In a random sample of 20 matched pairs, image dimensions were identical and 64-bit perceptual-hash distances were 0–3, although compressed bytes differed. This is strong evidence that MTSW is a re-export of D-Fire images, not an external dataset for the D-Fire model.

### AI For Mankind mirror: 11,487 images

The export has 8,045 train, 2,296 validation and 1,146 test pairs. Its `data.yaml` lists `Fire`, `Smoke`, `fire`, `smoke` as four distinct IDs, so they need a case-insensitive two-class mapping before scoring. It mixes 5-value YOLO boxes with **173 polygon label lines** (137 train, 25 validation, 11 test). `audit_roboflow.py` derives bounding rectangles from polygons for visualization. The [contact sheet](dataset/roboflow/aiformankind-fire-smoke-v1-yolov8/contact_sheet.jpg) shows plausible positives but the only empty test image and the only empty validation image both visibly contain fire and smoke. There are effectively no trustworthy negative test examples.

Of 730 source-name groups appearing across splits, 455 have a pair with perceptual-hash distance ≤4/64; 414 are hash-identical. Some same-name groups are unrelated images, which is why the 730-name count alone is not a duplicate count. The mirror also shares 215 near-duplicate source-name groups with Fire Project. Its README attributes the material to AI For Mankind/HPWREN and declares CC BY-NC-SA 4.0. The [original repository](https://github.com/aiformankind/wildfire-smoke-dataset) describes a smaller annotated wildfire-smoke release, so the provenance of all 11,487 exported images needs checking before use as a clean external benchmark.

### Fire Data Annotations: 3,284 images

The exported v5 `data.yaml` has **one** class, `Fire`; the source project browser's historical `0` class does not appear as a separate class in this export. There are 2,621 train and 663 test pairs; all 3,284 images have at least one annotation, no invalid label lines, and no generated augmentation. Three sampled test images in the [contact sheet](dataset/roboflow/fire-data-annotations-v5-yolov8/contact_sheet.jpg) have plausible flame boxes. Smoke is visible in some images but is deliberately unlabeled, so this is flame-only data. The export declares its license as Public Domain. No normalized filenames match D-Fire; that alone does not prove the pictures are absent from D-Fire or another model's training data.

## Evaluation decision

These exports can be read by the existing Python/Pillow pipeline directly from ZIP. They are **image collections**, so they cannot give time-to-first-smoke, temporal alert behavior, or false alert episodes per camera-hour. Continue using whole labeled FURG/CMU videos for those questions.

For the next image trial, start with a hand-reviewed, incident-grouped Fire Project subset for both classes and a separate Fire Data Annotations flame-positive subset. Include verified clear and confusing negative images from an independent source. Treat the AI For Mankind mirror as additional curated smoke examples after class and polygon normalization. Do not count MTSW as a new generalization test beyond D-Fire.

Reproduce the local audit with:

```powershell
python audit_roboflow.py dataset/roboflow/fire-project-fire-and-smoke-v8-yolov8.zip
python audit_roboflow.py dataset/roboflow/mtsw-fire-smoke-v5-yolov8.zip
python audit_roboflow.py dataset/roboflow/aiformankind-fire-smoke-v1-yolov8.zip
python audit_roboflow.py dataset/roboflow/fire-data-annotations-v5-yolov8.zip
```
