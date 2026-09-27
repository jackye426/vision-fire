# Waste-pit fire: rough two-angle synchronization trial

Run on 27 September 2026 using the existing 1 fps detector outputs for the [waste-pit video](https://www.youtube.com/watch?v=5NAkyEmC0IU). The user identified a dozer landmark at playback second **2** in angle 1 and **56** in angle 2. I treated those as the same instant, so aligned time `+n` means source seconds `2+n` and `56+n`. The [side-by-side stills](outputs/youtube/5NAkyEmC0IU_aligned_views.jpg) show the assumption and the fire progression. The source is an edited video, not two independently timestamped files; this is a simulation of simultaneous views. In angle 1, the displayed CCTV clock advances from 10:29:57 at playback 2 s to 10:30:09 at playback 5 s, so playback seconds must not be read as real camera seconds.

The [25-second synchronized D-FINE replay](outputs/youtube/5NAkyEmC0IU_aligned_dfine_h264.mp4) uses the same saved 1 fps predictions, held on 5 fps video frames. Orange and green boxes lie in manually selected fire/plume areas; thin grey boxes lie elsewhere. The areas were chosen **after viewing this positive clip**, only to show how static-object boxes change an alert calculation. They are not an independent test or a deployable detector. The [full result table](outputs/youtube/multiangle_waste_pit.csv) and [replay script](analyze_waste_pit_multiangle.py) preserve the exact rules.

## Flame: two views of the same event

The table uses boxes at score ≥0.25 whose centres fall in the reviewed fire areas. “Repeat” means overlapping same-class boxes on two consecutive sampled frames (IoU ≥0.20); the time is the second frame. “Cross-camera” means each view has at least one flame box in the current or previous two aligned seconds. All times are **video playback seconds after the proposed anchor**, not real-time detection delays.

| Model | Angle 1 first / repeat | Angle 2 first / repeat | First cross-camera flame candidate |
| --- | ---: | ---: | ---: |
| D-Fire YOLOv8n | +4 / +5 s | +2 / +3 s | +4 s |
| FireViewer D-FINE M | +3 / +4 s | +2 / +3 s | **+3 s** |
| FireViewer YOLO11-M | +4 / +5 s | +3 / +4 s | +4 s |

D-FINE supplies the earliest two-view flame candidate. In this clip, cross-camera confirmation is **not earlier than the earliest single-camera two-frame alert**: angle 2 repeats at +3 s for D-FINE and D-Fire, and +4 s for YOLO11-M. Its potential value is independent visual evidence. This one positive clip cannot measure how often that would reject false alarms.

The whole-frame calculation shows why box location matters. D-FINE and YOLO11-M both have a **repeated flame box at aligned +1 s in angle 2**, before the visible fire in that view; their boxes are on static signs and waste, not the flame. Restricting to the reviewed fire area moves angle 2's first repeat to +3 s for D-FINE and +4 s for YOLO11-M. A shared incident state should retain camera ID, class, score, time, location and track history; merely counting two model hits or two camera hits can confirm unrelated objects.

## Smoke: confirmation can cost time

| Model | Angle 1 smoke first / repeat | Angle 2 smoke first / repeat | Cross-camera smoke, 2 s lookback |
| --- | ---: | ---: | ---: |
| D-Fire YOLOv8n | +11 / +13 s | +16 / none in reviewed area | +16 s |
| FireViewer D-FINE M | +11 / +12 s | +6 / +7 s | +14 s |
| FireViewer YOLO11-M | +10 / +11 s | +16 / +42 s | +16 s |

D-FINE sees smoke in angle 2 before angle 1. The strict two-second coincidence rule waits until +14 s, later than either camera's first local repeat. With a five-second lookback, D-FINE's cross-camera candidate is +11 s, but a longer lookback also makes unrelated events easier to combine. For early warning, a sensible provisional alert can come from one well-tracked camera; evidence from the other view can raise confidence when it arrives. The two stages need separate timing and false-alert measurements.

Shifting the angle-2 anchor by ±2 playback seconds moves the fire-area flame cross-camera candidate to +3–4 s for D-FINE, +4 s for D-Fire, and +4–5 s for YOLO11-M. Smoke timing is more sensitive. Exact synchronization and a longer negative-footage trial are needed before measuring any real false-alert benefit.

Run `python analyze_waste_pit_multiangle.py` from the project root to regenerate the CSV and local visuals from `dataset/youtube/5NAkyEmC0IU.mp4` and `outputs/youtube/frame_results.csv`. The source video and generated visuals are kept out of Git; the script and compact CSV can be versioned.
