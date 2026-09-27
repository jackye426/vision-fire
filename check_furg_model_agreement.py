"""Check the proposed D-FINE + D-Fire flame gate on saved FURG frames.

This is an image-level stress check for the same-camera provisional tier, not a
multi-camera evaluation. FURG's frame flame labels are used only as a proxy;
co-located model boxes are not checked against the flame rectangles here.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from analyze_waste_pit_multiangle import iou


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "furg" / "frame_results.csv"
OUTPUT = ROOT / "outputs" / "furg" / "model_agreement_check.csv"
NEGATIVE_CLIPS = {"non_fire_patrolbot_onboard", "coolerbot"}
MODELS = ("dfine", "dfire_yolov8n")
THRESHOLD = 0.25
MATCH_IOU = 0.20


def flame_boxes(row: dict) -> list[list[int]]:
    return [item["box"] for item in json.loads(row["predictions"])
            if item["label"] == "flame" and float(item["score"]) >= THRESHOLD]


def any_overlap(a: list[list[int]], b: list[list[int]]) -> bool:
    return any(iou(x, y) >= MATCH_IOU for x in a for y in b)


def count_episodes(flags: list[bool]) -> int:
    return sum(flag and (index == 0 or not flags[index - 1])
               for index, flag in enumerate(flags))


def main() -> None:
    grouped = defaultdict(dict)
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in MODELS:
                key = row["clip"], int(row["frame"])
                if row["model"] in grouped[key]:
                    raise ValueError(f"Duplicate row {key}/{row['model']}")
                grouped[key][row["model"]] = row
    for key, models in grouped.items():
        if set(models) != set(MODELS):
            raise ValueError(f"Missing model at {key}")
    by_clip = defaultdict(list)
    for (clip, frame), models in grouped.items():
        labeled = bool(int(models["dfine"]["flame_labeled"]))
        if labeled != bool(int(models["dfire_yolov8n"]["flame_labeled"])):
            raise ValueError(f"Label mismatch at {(clip, frame)}")
        dfine = flame_boxes(models["dfine"])
        dfire = flame_boxes(models["dfire_yolov8n"])
        by_clip[clip].append({"frame": frame,
                              "time_s": float(models["dfine"]["time_s"]),
                              "labeled": labeled,
                              "dfine": dfine, "dfire": dfire,
                              "agreement": any_overlap(dfine, dfire)})
    summary = []
    for clip, frames in sorted(by_clip.items()):
        frames.sort(key=lambda row: row["frame"])
        negative = clip in NEGATIVE_CLIPS
        selected = [row for row in frames if negative or row["labeled"]]
        if not selected:
            continue
        dfine_repeat = []
        for index, row in enumerate(frames):
            previous = frames[index - 1] if index else None
            dfine_repeat.append(bool(previous and
                                     0.5 <= row["time_s"] - previous["time_s"] <= 1.5 and
                                     any_overlap(row["dfine"], previous["dfine"])))
        repeat_by_frame = {row["frame"]: flag for row, flag in zip(frames, dfine_repeat)}
        def first_time(predicate) -> str:
            return next((f"{row['time_s']:.3f}" for row in selected if predicate(row)), "")

        summary.append({
            "clip": clip,
            "cohort": "flame_free_clip" if negative else "labeled_flame_frames",
            "frames": len(selected),
            "dfine_flame_frames": sum(bool(row["dfine"]) for row in selected),
            "dfire_flame_frames": sum(bool(row["dfire"]) for row in selected),
            "same_frame_agreement_frames": sum(row["agreement"] for row in selected),
            "dfine_repeat_frames": sum(repeat_by_frame[row["frame"]] for row in selected),
            "agreement_episodes": count_episodes([row["agreement"] for row in selected]),
            "dfine_repeat_episodes": count_episodes(
                [repeat_by_frame[row["frame"]] for row in selected]),
            "first_labeled_time_s": "" if negative else f"{selected[0]['time_s']:.3f}",
            "first_dfine_hit_time_s": first_time(lambda row: bool(row["dfine"])),
            "first_agreement_time_s": first_time(lambda row: row["agreement"]),
            "first_dfine_repeat_time_s": first_time(
                lambda row: repeat_by_frame[row["frame"]]),
        })
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved {OUTPUT}")
    for cohort in ("flame_free_clip", "labeled_flame_frames"):
        rows = [row for row in summary if row["cohort"] == cohort]
        print(cohort, {field: sum(int(row[field]) for row in rows)
                       for field in ("frames", "dfine_flame_frames",
                                     "dfire_flame_frames", "same_frame_agreement_frames",
                                     "dfine_repeat_frames", "agreement_episodes",
                                     "dfine_repeat_episodes")})


if __name__ == "__main__":
    main()
