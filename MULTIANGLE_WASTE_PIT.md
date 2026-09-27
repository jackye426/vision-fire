# Waste-pit fire: two unsynchronized views

**Correction, 27 September 2026:** The two sections of the [edited waste-pit video](https://www.youtube.com/watch?v=5NAkyEmC0IU) play at different speeds. The original clip shows the same grabber drop at angle 1 **7 s** and angle 2 **59 s**, and the water cannon starting at **31 s** and **71 s**. Those two user-identified events yield the approximate visual map `angle 2 second = 55.5 + 0.5 × angle 1 second`; the earlier dozer estimate at angle 1 2 s maps to angle 2 56.5 s. Angle 2 therefore traverses this event at about twice the edited playback speed of angle 1. The previous fixed-offset timing and false-alert claims are withdrawn. The [new shared-clock trial](SHARED_TEMPORAL_WASTE_PIT.md) uses the corrected map with a ±1 second margin to test candidate rules on this one event.

The [side-by-side D-FINE video](outputs/youtube/5NAkyEmC0IU_aligned_dfine_h264.mp4) and [stills](outputs/youtube/5NAkyEmC0IU_aligned_views.jpg) now apply that speed correction. Their grabber and water-cannon rows are useful visual checks; frames between those anchors are approximate matches. Angle 1's CCTV overlay advances from 10:29:57 at video 2 s to 10:30:09 at video 5 s, demonstrating that at least that section is sped up relative to camera time. Angle 2 has no usable common capture clock in this copy, so the panels are **not clock-synchronized feeds**.

## What the saved detections show within each view

The table below uses saved 1 fps boxes at score ≥0.25 whose centers fall within manually reviewed fire/plume areas. A repeat is an overlapping same-class box on two successive sampled frames (IoU ≥0.20). Times are **source-video playback seconds within each angle**, so times from different angles must not be subtracted to calculate detection delay. The areas were selected *after* seeing this positive clip and are not an independently tested alert rule.

| Model | Angle 1 flame first / repeat | Angle 2 flame first / repeat | Angle 1 smoke first / repeat | Angle 2 smoke first / repeat |
| --- | ---: | ---: | ---: | ---: |
| D-Fire YOLOv8n | 6 / 7 s | 58 / 59 s | 13 / 15 s | 72 s / none |
| FireViewer D-FINE M | 5 / 6 s | 58 / 59 s | 13 / 14 s | 62 / 63 s |
| FireViewer YOLO11-M | 6 / 7 s | 59 / 60 s | 12 / 13 s | 72 / 98 s |

Within angle 2, whole-frame D-FINE flame boxes at 56–57 s repeatedly land on static clutter before visible flame. FireViewer YOLO11-M also produces early false flame boxes there. Persistence on one location can therefore preserve an error. At 58 s, a small flame is visible; D-FINE and D-Fire both box it in the same image with approximately 0.95 box overlap. This is useful **same-camera, two-model** evidence, independent of any cross-view timing assumption. The [source frame](outputs/youtube/5NAkyEmC0IU_source58.jpg) can be checked visually.

The [result table](outputs/youtube/multiangle_waste_pit.csv) contains the per-view source times only. The [script](analyze_waste_pit_multiangle.py) regenerates it and the local visuals from `dataset/youtube/5NAkyEmC0IU.mp4` and `outputs/youtube/frame_results.csv`. Those source and rendered media files stay local and are not included in Git.

## What remains unmeasured

These two views show why different perspectives *could* help: one camera may see a small flame while the other has a different obstruction or distractor. The shared-clock case study shows when candidate rules fire under the assumed margin, but cannot measure which live feed would alert first or how often combining them suppresses false alerts. That needs original simultaneous camera files with trustworthy capture timestamps or a validated time mapping across the whole period. It also needs multiple fires and substantial clear footage; this one edited positive clip cannot produce a false-alert rate.
