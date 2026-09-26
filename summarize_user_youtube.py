"""Summarize first detections and location-persistent alerts on user clips."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_user_youtube import CLIPS, MODELS, OUT
from detector import SPECS


EVENTS = (
    # (id, clip, class, segment start, user marker, segment end).
    # Markers are approximate video playback times supplied by the user.
    ("battery_pit_flame", "CI6YpclCYA4", "flame", 0, 6, 19),
    ("recycling_smoke", "WsUjSE-ibKo", "smoke", 0, 2, 93),
    ("recycling_flame", "WsUjSE-ibKo", "flame", 0, 10, 30),
    ("waste_pit_angle1_flame", "5NAkyEmC0IU", "flame", 0, 2, 52),
    ("waste_pit_angle1_smoke", "5NAkyEmC0IU", "smoke", 0, 10, 52),
    ("waste_pit_angle2_flame", "5NAkyEmC0IU", "flame", 53, 59, 102),
    ("waste_pit_angle2_smoke", "5NAkyEmC0IU", "smoke", 53, 61, 102),
)
VIDEO_TITLES = {
    "CI6YpclCYA4": "Fire caused by lithium-ion batteries",
    "WsUjSE-ibKo": "Lithium Ion battery fire at ecomaine's Recycling Facility",
    "ZBrOzQRoI-E": "CCTV footage of battery fires and explosions at waste facilities",
    "5NAkyEmC0IU": "Watch the Fire Rover Successfully Respond to a WtE Waste Pit Fire",
}


def iou(a: list[int], b: list[int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def candidates(row: dict, label: str, threshold: float) -> list[dict]:
    return [d for d in json.loads(row["detections"])
            if d["label"] == label and float(d["score"]) >= threshold]


def first_hit_and_alert(by_time: dict[int, dict], label: str, onset: int,
                        end: int, threshold: float) -> tuple[int | None, int | None]:
    first = None
    repeat = None
    for t in range(onset, end + 1):
        current = candidates(by_time[t], label, threshold)
        if current and first is None:
            first = t
        if t > onset and repeat is None:
            previous = candidates(by_time[t - 1], label, threshold)
            if any(iou(a["box"], b["box"]) >= 0.20 for a in current for b in previous):
                repeat = t  # Alert is emitted on the second sampled hit.
    return first, repeat


def main() -> None:
    with OUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = defaultdict(dict)
    for row in rows:
        key = (row["clip"], row["model"])
        time_s = int(row["time_s"])
        if time_s in data[key]:
            raise RuntimeError(f"Duplicate result {key} at {time_s}s")
        data[key][time_s] = row
    for clip in CLIPS:
        clip_times = {t for (c, _), rs in data.items() if c == clip for t in rs}
        for model in MODELS:
            if set(data[(clip, model)]) != clip_times:
                raise RuntimeError(f"Incomplete {clip}/{model}")
    event_rows = []
    for event_id, clip, label, segment_start, onset, end in EVENTS:
        for model in MODELS:
            applicable = label in SPECS[model]["labels"]
            for threshold in (0.25, 0.50):
                if applicable:
                    by_time = data[(clip, model)]
                    first, repeat = first_hit_and_alert(by_time, label, onset, end, threshold)
                    pre = sum(bool(candidates(by_time[t], label, threshold))
                              for t in range(segment_start, onset))
                else:
                    first = repeat = None
                    pre = None
                event_rows.append({
                    "event": event_id, "clip": clip, "class": label,
                    "segment_start_s": segment_start, "user_onset_s": onset,
                    "window_end_s": end, "model": model,
                    "score_threshold": threshold,
                    "first_hit_s": "" if first is None else first,
                    "first_location_repeat_s": "" if repeat is None else repeat,
                    "repeat_delay_from_user_marker_s": "" if repeat is None else repeat - onset,
                    "pre_marker_hit_frames": "" if pre is None else pre,
                    "applicable": int(applicable),
                })
    events_path = OUT.parent / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_rows[0])
        writer.writeheader()
        writer.writerows(event_rows)

    # The compilation changes scenes repeatedly, so keep it descriptive.
    compilation = []
    for model in MODELS:
        rs = data[("ZBrOzQRoI-E", model)]
        item = {"model": model, "sampled_frames": len(rs)}
        for label in ("smoke", "flame"):
            for threshold in (0.25, 0.50):
                item[f"{label}_{int(threshold * 100):03d}_frames"] = sum(
                    bool(candidates(row, label, threshold)) for row in rs.values())
        compilation.append(item)
    compilation_path = OUT.parent / "compilation.csv"
    with compilation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=compilation[0])
        writer.writeheader()
        writer.writerows(compilation)

    report = [
        "# User-selected YouTube fire and smoke clips",
        "",
        "Run on 26 September 2026. Four of six supplied videos were publicly retrievable at 720p; "
        "[K3ML54RtbAo](https://www.youtube.com/watch?v=K3ML54RtbAo) and "
        "[wm22sUiq9fE](https://www.youtube.com/watch?v=wm22sUiq9fE) returned ‘video not available’. "
        "The user subsequently supplied screen recordings of those two clips; their separate "
        "[review](../../SUPPLIED_CLIP_REPORT.md) uses recording playback time. "
        "The local files are under `dataset/youtube/`.",
        "",
        "| Clip | Length | User's approximate visible event time |",
        "| --- | ---: | --- |",
        "| [Battery fire](https://www.youtube.com/watch?v=CI6YpclCYA4) | 20 s | Flame ~6 s |",
        "| [ecomaine recycling fire](https://www.youtube.com/watch?v=WsUjSE-ibKo) | 94 s | Smoke ~2 s; flame ~10 s |",
        "| [CCTV compilation](https://www.youtube.com/watch?v=ZBrOzQRoI-E) | 42 s | Several separate shots; no single onset |",
        "| [Waste-pit fire, two angles](https://www.youtube.com/watch?v=5NAkyEmC0IU) | 103 s | First angle: flame ~2 s, smoke ~10–11 s; second angle: flame ~59 s, smoke ~61 s |",
        "",
        "All five local frame detectors were run on the same decoded frame at each integer video second "
        "(one sampled frame per second). At score ≥0.25, a location-persistent alert requires two consecutive "
        "sampled frames with same-class boxes overlapping at IoU ≥0.20; the timestamp is the second frame. "
        "A dash means no such alert within that event window; N/A means the model has no output for that class. "
        "Onset times are the user's approximate playback markers, not verified frame-level labels. "
        "Some videos are edited or accelerated, so playback seconds are not camera elapsed time.",
        "",
        "## Unverified location-repeat timings",
        "",
        "| Model | CI flame (6 s) | Ws smoke (2 s) | Ws flame (10 s) | 5NA A1 flame (2 s) | 5NA A1 smoke (10 s) | 5NA A2 flame (59 s) | 5NA A2 smoke (61 s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    short = {"dfire_yolov8n": "D-Fire YOLOv8n", "dfine": "D-FINE M", "yolo": "YOLO11-M",
             "soul": "SoulPerforms fire only", "pyronear": "PyroNear smoke only"}
    for model in MODELS:
        values = []
        for event_id, _, label, _, onset, _ in EVENTS:
            row = next(r for r in event_rows if r["event"] == event_id and r["model"] == model
                       and r["score_threshold"] == 0.25)
            if not int(row["applicable"]):
                values.append("N/A")
            elif row["first_location_repeat_s"] == "":
                values.append("—")
            else:
                t = int(row["first_location_repeat_s"])
                values.append(f"{t}s (+{t - onset}s)")
        report.append(f"| {short[model]} | " + " | ".join(values) + " |")
    report += [
        "",
        "The [frame-level results](frame_results.csv) and [event timings](events.csv) also contain score ≥0.50 "
        "decisions, first single-frame hits and pre-marker detections. The compilation's "
        "[detection counts](compilation.csv) describe model activity; they are not accuracy scores because "
        "the shots have no published frame labels and can include scene changes.",
        "",
        "## Visual check of early boxes",
        "",
        "The timing table reports repeated boxes anywhere in the frame; it is **not yet a verified "
        "event-detection table**. On the [battery-fire clip](https://www.youtube.com/watch?v=CI6YpclCYA4), "
        "D-Fire's 7 s repeat boxes the yellow excavator and SoulPerforms' 7 s repeat boxes the "
        "wall/door; neither marks the flame. After restricting this one event to the visually reviewed "
        "fire area (box center x < 400, y > 450 in the 1274×720 frame), D-Fire's first repeat is "
        "14 s, SoulPerforms' is 10 s, YOLO11-M's is 10 s, and D-FINE has one correct hit at 14 s "
        "but no repeat. This area check is a case-study correction, not a ground-truth benchmark. "
        "The [model review sheet](CI6YpclCYA4_model_review.jpg) shows the boxes.",
        "",
        "The [recycling review sheet](WsUjSE-ibKo_model_review.jpg) shows boxes on the visible "
        "smoke and small flames. The [waste-pit review sheet](5NAkyEmC0IU_model_review.jpg) shows "
        "D-FINE detecting small flames early, along with some boxes on unrelated clutter; smoke "
        "becomes much more conspicuous later. PyroNear's distant-wildfire-smoke detector makes no "
        "two-frame smoke alert on these indoor/recycling clips at this threshold and sampling rate. "
        "The [compilation review sheet](ZBrOzQRoI-E_model_review.jpg) shows why scene cuts need "
        "separate event labels.",
        "",
        "D-Fire's boxes can also be watched on the silent, 5 fps H.264 renders of the "
        "[battery-fire clip](CI6YpclCYA4_dfire_5fps_h264.mp4) and "
        "[recycling clip](WsUjSE-ibKo_dfire_5fps_h264.mp4). These are visual demos at score 0.25; "
        "the common five-model comparison above samples 1 fps. The renders retain playback speed "
        "but omit audio.",
        "The [recording index](../../WATCH_MODEL_VIDEOS.md) has side-by-side and individual "
        "videos for every frame model on all four clips. Those comparison videos play at 5 fps "
        "while holding the common 1 fps detections until the next sample.",
        "",
    ]
    temporal_path = OUT.parent / "pyronear_temporal_windows.csv"
    if temporal_path.exists():
        with temporal_path.open(newline="", encoding="utf-8") as handle:
            temporal_rows = list(csv.DictReader(handle))
        report += [
            "## Released PyroNear video pipeline",
            "",
            "The separate [PyroNear temporal package](https://huggingface.co/pyronear/temporal-model) "
            "was also run at 1 fps on three clips. It uses its own companion YOLO, smoke tubes and "
            "image-encoder/transformer classifier with the package's 0.346 decision threshold. "
            "Twenty-frame windows overlap by 10 frames, with a final tail window; the two waste-pit "
            "angles stay separate. The compilation was excluded because frequent shot cuts would "
            "break location tubes. A positive window is a **whole-window verdict**, not a first-alert "
            "timestamp or an ablation against the different PyroNear Sensitive checkpoint above.",
            "",
            "| Clip/segment | Positive windows | First positive window | Highest tube probability | Visual check |",
            "| --- | ---: | --- | ---: | --- |",
        ]
        segments = (
            ("CI6YpclCYA4", "battery_fire", "Battery fire", "False: tube stays on a yellow foreground bollard"),
            ("WsUjSE-ibKo", "recycling", "Recycling", "Smoke plume at ~17–20 s; much later than the ~2 s marker"),
            ("5NAkyEmC0IU", "angle1", "Waste pit angle 1", "Later upper-left flame/smoke region; early pit smoke missed"),
            ("5NAkyEmC0IU", "angle2", "Waste pit angle 2", "No positive window"),
        )
        for clip, segment, title, note in segments:
            rows = [r for r in temporal_rows if r["clip"] == clip and r["segment"] == segment]
            positives = [r for r in rows if int(r["smoke_decision"])]
            first = (f"{positives[0]['window_start_s']}–{positives[0]['window_end_s']} s"
                     if positives else "—")
            maximum = max((float(r["max_tube_probability"]) for r in rows), default=0.0)
            report.append(f"| {title} | {len(positives)}/{len(rows)} | {first} | {maximum:.3f} | {note} |")
        report += [
            "",
            "The [window decisions](pyronear_temporal_windows.csv), "
            "[candidate positions and tube details](pyronear_candidate_details.json), and "
            "[visual review sheet](pyronear_candidate_review.jpg) preserve this check. The battery "
            "false positive is a direct example of a temporal model accepting a persistent static "
            "distractor. The recycling detection is real smoke, while its companion detector stops "
            "proposing boxes once the scene fills with smoke.",
            "",
        ]
    report += [
        "These are four positive, edited examples, with no long clear-camera period. They are useful case studies "
        "for small flames, smoke, camera angle changes and alert persistence; they cannot establish false-alert "
        "episodes per camera-hour or a general model ranking. Review the actual boxes when a model fires before "
        "the stated onset or misses a visible event.",
        "",
        "Reproduce the common 1 fps run from the downloaded source files with "
        "`python benchmark_user_youtube.py` and `python summarize_user_youtube.py`. "
        "Render one D-Fire demo with `python render_dfire_youtube.py WsUjSE-ibKo`; "
        "transcode its MP4 to H.264 with ffmpeg for browser playback. Run the temporal system with "
        "`.\\.venv-temporal\\Scripts\\python.exe benchmark_pyronear_temporal_youtube.py`. "
        "The full pedbrgs method was not run on these clips.",
    ]
    (OUT.parent / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved {events_path}, {compilation_path} and {OUT.parent / 'REPORT.md'}")


if __name__ == "__main__":
    main()
