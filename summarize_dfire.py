"""Summarise reproducible D-Fire image and video benchmark CSVs."""

import csv
import statistics
from pathlib import Path

from detector import SPECS


OUT = Path(__file__).parent / "outputs" / "dfire_benchmark"
THRESHOLDS = ("025", "050")


def read_rows(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_streak(rows: list[dict], column: str, required: int):
    streak = 0
    for row in rows:
        streak = streak + 1 if int(row[column]) else 0
        if streak == required:
            return row["time_s"]
    return ""


def episodes(rows: list[dict], column: str, required: int) -> int:
    streak = 0
    total = 0
    for row in rows:
        streak = streak + 1 if int(row[column]) else 0
        if streak == required:
            total += 1
    return total


def image_summary(rows: list[dict]) -> list[dict]:
    output = []
    for model in SPECS:
        group = [row for row in rows if row["model"] == model]
        if len(group) != 160:
            raise RuntimeError(f"Incomplete image results for {model}: {len(group)}")
        for threshold in THRESHOLDS:
            clear = [row for row in group if row["category"] == "none"]
            clear_hits = sum(
                int(row[f"smoke_{threshold}"]) or int(row[f"flame_{threshold}"])
                for row in clear
            )
            for label, positive_categories in (
                ("smoke", {"smoke_only", "both"}),
                ("flame", {"flame_only", "both"}),
            ):
                if label not in SPECS[model]["labels"]:
                    continue
                positives = [row for row in group if row["category"] in positive_categories]
                negatives = [row for row in group if row["category"] not in positive_categories]
                column = f"{label}_{threshold}"
                tp = sum(int(row[column]) for row in positives)
                fp = sum(int(row[column]) for row in negatives)
                output.append({
                    "model": model, "threshold": f"0.{threshold[1:]}", "class": label,
                    "tp": tp, "fn": len(positives) - tp,
                    "fp": fp, "tn": len(negatives) - fp,
                    "recall": f"{tp / len(positives):.3f}",
                    "false_positive_rate": f"{fp / len(negatives):.3f}",
                    "clear_image_hits": clear_hits,
                    "clear_images": len(clear),
                    "median_cpu_ms": f"{statistics.median(float(row['latency_ms']) for row in group):.1f}",
                })
    return output


def video_summary(rows: list[dict]) -> list[dict]:
    output = []
    clips = sorted({row["video"] for row in rows})
    for clip in clips:
        for model in SPECS:
            group = sorted((row for row in rows if row["video"] == clip and row["model"] == model),
                           key=lambda row: int(row["frame"]))
            if not group:
                raise RuntimeError(f"Missing video results for {clip} / {model}")
            for threshold in THRESHOLDS:
                any_rows = []
                for row in group:
                    copy = dict(row)
                    copy["any"] = int(row[f"smoke_{threshold}"]) or int(row[f"flame_{threshold}"])
                    any_rows.append(copy)
                output.append({
                    "video": clip, "model": model, "threshold": f"0.{threshold[1:]}",
                    "sampled_frames": len(group),
                    "flagged_frames": sum(row["any"] for row in any_rows),
                    "first_raw_s": first_streak(any_rows, "any", 1),
                    "first_2_hit_s": first_streak(any_rows, "any", 2),
                    "first_smoke_s": first_streak(group, f"smoke_{threshold}", 1),
                    "first_flame_s": first_streak(group, f"flame_{threshold}", 1),
                    "alert_episodes_2_hit": episodes(any_rows, "any", 2),
                    "median_cpu_ms": f"{statistics.median(float(row['latency_ms']) for row in group):.1f}",
                })
    return output


def write_rows(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main() -> None:
    write_rows("image_summary.csv", image_summary(read_rows("image_results.csv")))
    write_rows("video_summary.csv", video_summary(read_rows("video_frames.csv")))


if __name__ == "__main__":
    main()
