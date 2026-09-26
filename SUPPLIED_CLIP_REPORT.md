# Two user-supplied screen recordings

Run on 26 September 2026. The user supplied two Windows screen recordings corresponding to previously inaccessible YouTube links:

| Screen recording | Source link | Length | Local SHA-256 |
| --- | --- | ---: | --- |
| `20260926-1156-27.7958810.mp4` | [K3ML54RtbAo](https://www.youtube.com/watch?v=K3ML54RtbAo) | 38.57 s | `5fb17c2036bd8f97c41b735cae63e11f8050e1f07d1c124c2b758b0647d14445` |
| `20260926-1158-19.5475455.mp4` | [wm22sUiq9fE](https://www.youtube.com/watch?v=wm22sUiq9fE) | 26.27 s | `7be67576b98a09a0016e1679bd699b90c527d41576f64579bef4d9678037a030` |

The recordings were copied byte-for-byte to `dataset/youtube/` under their corresponding video IDs. Results use **recording playback seconds**. The K3ML recording starts with an established flame already visible, so it cannot measure delay from the source video's first flame. The wm22 recording includes occasional YouTube player overlays; its small flame appears around recording 16–18 s, with a pale plume beneath the raised loader bucket afterward. These observations are visual reviews, not frame-level labels from the source publisher.

The five local frame checkpoints ran on each integer recording second with their normal input size and a model-output score cutoff of 0.25. A location-repeat below means same-class boxes in two consecutive sampled seconds overlap at IoU ≥0.20. That rule does **not** establish that the box covers the hazard. PyroNear's separately released video pipeline was run in 20-frame windows at 1 fps, with its packaged decision threshold.

## K3ML: flame already visible at recording start

| Model | First flame box | First overlapping repeat | Visual check |
| --- | ---: | ---: | --- |
| D-Fire YOLOv8n | 0 s | 1 s | On visible flame |
| D-FINE M | 0 s | 1 s | On visible flame; some smoke boxes also cover dark ceiling/structure |
| FireViewer YOLO11-M | 0 s | 1 s | On visible flame |
| SoulPerforms fire only | 0 s | 1 s | On visible flame initially; later boxes unrelated bright regions after suppression |
| PyroNear Sensitive smoke only | Not applicable | Not applicable | No flame output by design |

The PyroNear temporal pipeline was negative on 0–19 s and 10–29 s windows, then positive on 19–38 s (`p=0.605`). Its companion detector's boxes at 27–36 s stay on the bright **right-edge floor/structure**, away from the fire and visible plume. The positive window is therefore a false location, not a successful smoke alert.

## wm22: small flame under a loader bucket

| Model | Flame result at score 0.25 | Smoke result | Visual check |
| --- | --- | --- | --- |
| D-Fire YOLOv8n | No flame box | One isolated smoke box at 5 s | Missed the small flame |
| D-FINE M | One flame hit at 16 s; no repeat | Repeated smoke boxes at 20–21 s | Flame box lands on the visible flame; smoke boxes cover the pale plume area |
| FireViewer YOLO11-M | One flame hit at 17 s; no repeat | Repeated smoke boxes at 20–21 s | Flame box lands on the visible flame; smoke boxes cover the pale plume area |
| SoulPerforms fire only | Many repeated flame-class boxes | No smoke output by design | Early/repeated boxes land on loader parts, rather than the small flame |
| PyroNear Sensitive smoke only | No flame output by design | Repeated smoke boxes at 24–25 s | Boxes land on a static pale wall mark, not the plume |

The PyroNear temporal companion detector proposed no boxes and both video windows were smoke-negative. The true flame hits from D-FINE and YOLO11-M are **single sampled frames**; a strict two-frame alert rule would miss them. Their smoke boxes at 20–21 s are visually plausible, but the plume is faint and source ground truth is unavailable.

[Watch both recordings with all models side by side, or open any model separately](WATCH_MODEL_VIDEOS.md). The videos play the source at 5 fps; the shared benchmark boxes update at 1 fps and are held between samples. The temporal panel shows yellow companion proposals and only reveals whole-window decisions when each window completes. The [frame CSV](outputs/youtube/frame_results.csv), [temporal-window CSV](outputs/youtube/pyronear_temporal_windows.csv), and [companion-box CSV](outputs/youtube/pyronear_companion_frames.csv) preserve the raw outputs.

These two short positive screen recordings show useful failure modes but cannot establish a general model ranking or false alerts per camera-hour. In particular, the source videos may have edits and the screen recording clock does not necessarily match the original video timeline.
