"""Summarize the exploratory FURG flame-video transfer check."""

import csv
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent / "outputs" / "furg"
SOURCE = ROOT / "frame_results.csv"
SUMMARY = ROOT / "summary.csv"
CLIPS = ROOT / "clip_summary.csv"
CLEAR_CLIPS = {"non_fire_patrolbot_onboard", "coolerbot"}
MODELS = ("dfire_yolov8n", "dfine", "yolo", "soul", "pyronear")


def first_two(flags: list[bool]) -> float | None:
    for index in range(1, len(flags)):
        if flags[index - 1] and flags[index]:
            return index
    return None


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    summaries = []
    clips = []
    for threshold in (0.25, 0.50):
        suffix = f"{int(threshold * 100):03d}"
        for model in MODELS:
            rows = [r for r in records if r["model"] == model]
            positives = [r for r in rows if int(r["flame_labeled"])]
            negatives = [r for r in rows if r["clip"] in CLEAR_CLIPS]
            clear_two_hit = 0
            for clip in CLEAR_CLIPS:
                group = sorted((r for r in rows if r["clip"] == clip),
                               key=lambda r: int(r["frame"]))
                clear_two_hit += int(first_two([bool(int(r[f"flame_{suffix}"]))
                                                for r in group]) is not None)
            summaries.append({
                "model": model, "threshold": threshold,
                "flame_labeled_frames": len(positives),
                "flame_hit_frames": sum(int(r[f"flame_{suffix}"]) for r in positives),
                "flame_aligned_iou50_frames": sum(int(r[f"flame_iou50_{suffix}"])
                                                  for r in positives),
                "clear_clip_frames": len(negatives),
                "clear_clip_flame_flags": sum(int(r[f"flame_{suffix}"]) for r in negatives),
                "clear_clips_with_2_hit_flame": clear_two_hit,
                "clear_clip_smoke_flags_unscored": sum(int(r[f"smoke_{suffix}"])
                                                       for r in negatives),
                "median_cpu_ms_per_frame": round(statistics.median(float(r["latency_ms"])
                                                                     for r in rows), 1),
            })
            for clip in sorted({r["clip"] for r in rows}):
                group = sorted((r for r in rows if r["clip"] == clip),
                               key=lambda r: int(r["frame"]))
                gt = next((float(r["time_s"]) for r in group
                           if int(r["flame_labeled"])), None)
                first_model = next((float(r["time_s"]) for r in group
                                    if int(r[f"flame_{suffix}"])), None)
                two_index = first_two([bool(int(r[f"flame_{suffix}"])) for r in group])
                hits = sum(int(r[f"flame_{suffix}"]) for r in group
                           if int(r["flame_labeled"]))
                labeled = sum(int(r["flame_labeled"]) for r in group)
                clips.append({
                    "clip": clip, "model": model, "threshold": threshold,
                    "sampled_frames": len(group), "flame_labeled_frames": labeled,
                    "first_label_s": gt if gt is not None else "",
                    "first_model_flame_s": first_model if first_model is not None else "",
                    "first_two_hit_flame_s": (float(group[two_index]["time_s"])
                                              if two_index is not None else ""),
                    "flame_hit_frames": hits,
                })
    save(SUMMARY, summaries)
    save(CLIPS, clips)
    for row in summaries:
        if row["threshold"] == 0.25:
            print(row)
    print(f"Saved {SUMMARY} and {CLIPS}")


if __name__ == "__main__":
    main()
