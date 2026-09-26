"""Aggregate clip-level CMU smoke decisions, keeping temporal setup separate."""

import csv
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent / "outputs" / "cmu"


def read(name: str) -> list[dict]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    clips = read("clip_summary.csv")
    rows = []
    for threshold in ("0.25", "0.5"):
        for model in ("dfire_yolov8n", "dfine", "yolo", "soul", "pyronear"):
            group = [r for r in clips if r["threshold"] == threshold
                     and r["model"] == model]
            positive = [r for r in group if r["label"] == "smoke"]
            clear = [r for r in group if r["label"] == "clear"]
            rows.append({
                "model": model, "threshold": threshold,
                "positive_clips_with_smoke_2hit_of_3": sum(int(r["smoke_two_consecutive"])
                                                          for r in positive),
                "clear_clips_with_smoke_2hit_of_3": sum(int(r["smoke_two_consecutive"])
                                                       for r in clear),
                "positive_smoke_flagged_frames_of_108": sum(int(r["smoke_flagged_frames"])
                                                             for r in positive),
                "clear_smoke_flagged_frames_of_108": sum(int(r["smoke_flagged_frames"])
                                                          for r in clear),
                "median_cpu_ms_per_frame": round(statistics.median(float(r["median_cpu_ms"])
                                                                     for r in group), 1),
            })
    target = ROOT / "aggregate.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["threshold"] == "0.25":
            print(row)
    temporal = read("pyronear_temporal_windows.csv")
    for label in ("smoke", "clear"):
        clips_for_label = {r["clip"] for r in temporal if r["label"] == label}
        flagged = sum(any(int(r["smoke_decision"]) for r in temporal
                          if r["clip"] == clip) for clip in clips_for_label)
        print(f"PyroNear temporal {label}: {flagged}/{len(clips_for_label)} clips flagged")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
