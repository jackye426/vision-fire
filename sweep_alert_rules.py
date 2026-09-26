"""Replay existing D-Fire frame scores through class-specific alert rules.

No neural network inference is repeated. These candidate times are relative to
clip start; the public clips have no frame-level event/onset ground truth.
"""

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs" / "dfire_benchmark"
SOURCE = ROOT / "video_frames.csv"
TARGET = ROOT / "alert_rule_sweep.csv"
THRESHOLDS = (0.25, 0.50)
RULES = ((1, 1), (2, 2), (3, 3), (2, 3))  # required hits, most recent sampled frames


def flag_series(scores: list[float], threshold: float, hits: int, span: int) -> list[bool]:
    flags = []
    for index in range(len(scores)):
        recent = scores[max(0, index - span + 1):index + 1]
        flags.append(len(recent) >= hits and sum(value >= threshold for value in recent) >= hits)
    return flags


def main() -> None:
    groups = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[(row["video"], row["model"])].append(row)
    rows = []
    for (video, model), frames in sorted(groups.items()):
        frames.sort(key=lambda row: int(row["frame"]))
        times = [float(row["time_s"]) for row in frames]
        scores = {
            "smoke": [float(row["max_smoke"]) for row in frames],
            "flame": [float(row["max_flame"]) for row in frames],
        }
        for threshold in THRESHOLDS:
            for hits, span in RULES:
                class_flags = {name: flag_series(values, threshold, hits, span)
                               for name, values in scores.items()}
                alerts = [class_flags["smoke"][index] or class_flags["flame"][index]
                          for index in range(len(frames))]
                first_index = next((index for index, alert in enumerate(alerts) if alert), None)
                episodes = sum(alert and (index == 0 or not alerts[index - 1])
                               for index, alert in enumerate(alerts))
                rows.append({
                    "video": video, "model": model, "threshold": threshold,
                    "rule": f"{hits}_of_{span}_same_class", "sampled_frames": len(frames),
                    "first_alert_s": times[first_index] if first_index is not None else "",
                    "flagged_sampled_frames": sum(alerts), "alert_episodes": episodes,
                })
    with TARGET.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} model/clip/threshold/rule rows to {TARGET}")
    for video in ("FP31.mp4", "VP3.mp4"):
        print(f"{video}, threshold 0.25:")
        for row in rows:
            if row["video"] == video and row["threshold"] == 0.25:
                print(f"  {row['model']:10s} {row['rule']:18s} first={row['first_alert_s']!s:>4s} "
                      f"episodes={row['alert_episodes']}")


if __name__ == "__main__":
    main()
