# User-selected YouTube fire and smoke clips

Run on 26 September 2026. Four of six supplied videos were publicly retrievable at 720p; [K3ML54RtbAo](https://www.youtube.com/watch?v=K3ML54RtbAo) and [wm22sUiq9fE](https://www.youtube.com/watch?v=wm22sUiq9fE) returned ‘video not available’. The user subsequently supplied screen recordings of those two clips; their separate [review](../../SUPPLIED_CLIP_REPORT.md) uses recording playback time. The local files are under `dataset/youtube/`.

| Clip | Length | User's approximate visible event time |
| --- | ---: | --- |
| [Battery fire](https://www.youtube.com/watch?v=CI6YpclCYA4) | 20 s | Flame ~6 s |
| [ecomaine recycling fire](https://www.youtube.com/watch?v=WsUjSE-ibKo) | 94 s | Smoke ~2 s; flame ~10 s |
| [CCTV compilation](https://www.youtube.com/watch?v=ZBrOzQRoI-E) | 42 s | Several separate shots; no single onset |
| [Waste-pit fire, two angles](https://www.youtube.com/watch?v=5NAkyEmC0IU) | 103 s | First angle: flame ~2 s, smoke ~10–11 s; second angle: flame ~59 s, smoke ~61 s |

All five local frame detectors were run on the same decoded frame at each integer video second (one sampled frame per second). At score ≥0.25, a location-persistent alert requires two consecutive sampled frames with same-class boxes overlapping at IoU ≥0.20; the timestamp is the second frame. A dash means no such alert within that event window; N/A means the model has no output for that class. Onset times are the user's approximate playback markers, not verified frame-level labels. Some videos are edited or accelerated, so playback seconds are not camera elapsed time.

## Unverified location-repeat timings

| Model | CI flame (6 s) | Ws smoke (2 s) | Ws flame (10 s) | 5NA A1 flame (2 s) | 5NA A1 smoke (10 s) | 5NA A2 flame (59 s) | 5NA A2 smoke (61 s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-Fire YOLOv8n | 7s (+1s) | 8s (+6s) | 11s (+1s) | 7s (+5s) | 15s (+5s) | 60s (+1s) | 78s (+17s) |
| D-FINE M | — | 7s (+5s) | 11s (+1s) | 6s (+4s) | 14s (+4s) | 60s (+1s) | 63s (+2s) |
| YOLO11-M | 10s (+4s) | 11s (+9s) | 11s (+1s) | 7s (+5s) | 13s (+3s) | 60s (+1s) | 98s (+37s) |
| SoulPerforms fire only | 7s (+1s) | N/A | 12s (+2s) | 13s (+11s) | N/A | 65s (+6s) | N/A |
| PyroNear smoke only | N/A | — | N/A | N/A | — | N/A | — |

The [frame-level results](frame_results.csv) and [event timings](events.csv) also contain score ≥0.50 decisions, first single-frame hits and pre-marker detections. The compilation's [detection counts](compilation.csv) describe model activity; they are not accuracy scores because the shots have no published frame labels and can include scene changes.

## Visual check of early boxes

The timing table reports repeated boxes anywhere in the frame; it is **not yet a verified event-detection table**. On the [battery-fire clip](https://www.youtube.com/watch?v=CI6YpclCYA4), D-Fire's 7 s repeat boxes the yellow excavator and SoulPerforms' 7 s repeat boxes the wall/door; neither marks the flame. After restricting this one event to the visually reviewed fire area (box center x < 400, y > 450 in the 1274×720 frame), D-Fire's first repeat is 14 s, SoulPerforms' is 10 s, YOLO11-M's is 10 s, and D-FINE has one correct hit at 14 s but no repeat. This area check is a case-study correction, not a ground-truth benchmark. The [model review sheet](CI6YpclCYA4_model_review.jpg) shows the boxes.

The [recycling review sheet](WsUjSE-ibKo_model_review.jpg) shows boxes on the visible smoke and small flames. The [waste-pit review sheet](5NAkyEmC0IU_model_review.jpg) shows D-FINE detecting small flames early, along with some boxes on unrelated clutter; smoke becomes much more conspicuous later. PyroNear's distant-wildfire-smoke detector makes no two-frame smoke alert on these indoor/recycling clips at this threshold and sampling rate. The [compilation review sheet](ZBrOzQRoI-E_model_review.jpg) shows why scene cuts need separate event labels.

D-Fire's boxes can also be watched on the silent, 5 fps H.264 renders of the [battery-fire clip](CI6YpclCYA4_dfire_5fps_h264.mp4) and [recycling clip](WsUjSE-ibKo_dfire_5fps_h264.mp4). These are visual demos at score 0.25; the common five-model comparison above samples 1 fps. The renders retain playback speed but omit audio.
The [recording index](../../WATCH_MODEL_VIDEOS.md) has side-by-side and individual videos for every frame model on all four clips. Those comparison videos play at 5 fps while holding the common 1 fps detections until the next sample.

## Released PyroNear video pipeline

The separate [PyroNear temporal package](https://huggingface.co/pyronear/temporal-model) was also run at 1 fps on three clips. It uses its own companion YOLO, smoke tubes and image-encoder/transformer classifier with the package's 0.346 decision threshold. Twenty-frame windows overlap by 10 frames, with a final tail window; the two waste-pit angles stay separate. The compilation was excluded because frequent shot cuts would break location tubes. A positive window is a **whole-window verdict**, not a first-alert timestamp or an ablation against the different PyroNear Sensitive checkpoint above.

| Clip/segment | Positive windows | First positive window | Highest tube probability | Visual check |
| --- | ---: | --- | ---: | --- |
| Battery fire | 1/1 | 0–19 s | 0.936 | False: tube stays on a yellow foreground bollard |
| Recycling | 2/9 | 0–19 s | 0.887 | Smoke plume at ~17–20 s; much later than the ~2 s marker |
| Waste pit angle 1 | 2/5 | 30–49 s | 0.792 | Later upper-left flame/smoke region; early pit smoke missed |
| Waste pit angle 2 | 0/4 | — | 0.000 | No positive window |

The [window decisions](pyronear_temporal_windows.csv), [candidate positions and tube details](pyronear_candidate_details.json), and [visual review sheet](pyronear_candidate_review.jpg) preserve this check. The battery false positive is a direct example of a temporal model accepting a persistent static distractor. The recycling detection is real smoke, while its companion detector stops proposing boxes once the scene fills with smoke.

These are four positive, edited examples, with no long clear-camera period. They are useful case studies for small flames, smoke, camera angle changes and alert persistence; they cannot establish false-alert episodes per camera-hour or a general model ranking. Review the actual boxes when a model fires before the stated onset or misses a visible event.

Reproduce the common 1 fps run from the downloaded source files with `python benchmark_user_youtube.py` and `python summarize_user_youtube.py`. Render one D-Fire demo with `python render_dfire_youtube.py WsUjSE-ibKo`; transcode its MP4 to H.264 with ffmpeg for browser playback. Run the temporal system with `.\.venv-temporal\Scripts\python.exe benchmark_pyronear_temporal_youtube.py`. The full pedbrgs method was not run on these clips.
