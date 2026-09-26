"""Compare a smoke specialist + flame specialist against dual-class models.

This is a decision-level OR: PyroNear Sensitive supplies smoke candidates and
SoulPerforms supplies flame candidates. It is not a trained fusion model.
"""

import csv
import statistics
from pathlib import Path


OUT = Path(__file__).parent / "outputs" / "dfire_benchmark"
PAIR = ("pyronear", "soul")
THRESHOLDS = (0.25, 0.50)


def read(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_two(times: list[float], flags: list[bool]):
    for index in range(1, len(flags)):
        if flags[index - 1] and flags[index]:
            return times[index]
    return ""


def main() -> None:
    images = read("image_results.csv")
    by_image = {(r["name"], r["model"]): r for r in images}
    names = sorted({r["name"] for r in images})
    localization = read("localization_results.csv")
    loc_count = {
        "smoke": sum(int(r["localized_iou50"]) for r in localization
                     if r["model"] == "pyronear" and r["class"] == "smoke"),
        "flame": sum(int(r["localized_iou50"]) for r in localization
                     if r["model"] == "soul" and r["class"] == "flame"),
    }
    image_rows = []
    for threshold in THRESHOLDS:
        smoke_hits = flame_hits = clear_hits = 0
        latencies = []
        for name in names:
            smoke = by_image[(name, "pyronear")]
            flame = by_image[(name, "soul")]
            category = smoke["category"]
            smoke_hit = float(smoke["max_smoke"]) >= threshold
            flame_hit = float(flame["max_flame"]) >= threshold
            smoke_hits += int(category in ("smoke_only", "both") and smoke_hit)
            flame_hits += int(category in ("flame_only", "both") and flame_hit)
            clear_hits += int(category == "none" and (smoke_hit or flame_hit))
            latencies.append(float(smoke["latency_ms"]) + float(flame["latency_ms"]))
        image_rows.append({
            "combination": "pyronear_smoke_OR_soul_flame", "threshold": threshold,
            "smoke_hits_of_80": smoke_hits, "flame_hits_of_80": flame_hits,
            "clear_flags_of_40": clear_hits,
            "smoke_aligned_iou50_of_80_at_025": loc_count["smoke"] if threshold == 0.25 else "",
            "flame_aligned_iou50_of_80_at_025": loc_count["flame"] if threshold == 0.25 else "",
            "median_cpu_ms_per_image": round(statistics.median(latencies), 1),
        })
    write("specialist_pair_image.csv", image_rows)

    frames = read("video_frames.csv")
    by_frame = {(r["video"], int(r["frame"]), r["model"]): r for r in frames}
    video_rows = []
    for video in sorted({r["video"] for r in frames}):
        indices = sorted({int(r["frame"]) for r in frames if r["video"] == video})
        for threshold in THRESHOLDS:
            times = []
            flags = []
            smoke_flags = []
            flame_flags = []
            for index in indices:
                smoke = by_frame[(video, index, "pyronear")]
                flame = by_frame[(video, index, "soul")]
                s = float(smoke["max_smoke"]) >= threshold
                f = float(flame["max_flame"]) >= threshold
                times.append(float(smoke["time_s"]))
                flags.append(s or f)
                smoke_flags.append(s)
                flame_flags.append(f)
            video_rows.append({
                "video": video, "threshold": threshold, "sampled_frames": len(indices),
                "flagged_frames": sum(flags), "smoke_flagged_frames": sum(smoke_flags),
                "flame_flagged_frames": sum(flame_flags),
                "first_2_hit_s": first_two(times, flags),
            })
    write("specialist_pair_video.csv", video_rows)
    for row in image_rows:
        print(row)
    for row in video_rows:
        if row["threshold"] == 0.25:
            print(row)


if __name__ == "__main__":
    main()
