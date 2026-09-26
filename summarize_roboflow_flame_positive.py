"""Summarize the source-distinct, positive-only Roboflow flame trial."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from benchmark_roboflow_flame_positive import MODELS, OUT, clean_test_images, ARCHIVE
import zipfile


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        images, cohort = clean_test_images(archive.namelist())
    selected = set(images)
    with (OUT / "per_image.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["image"] in selected]
    keyed = {(row["image"], row["model"]): row for row in rows}
    expected = {(image, model) for image in images for model in MODELS}
    if len(keyed) != len(expected) or set(keyed) != expected:
        raise RuntimeError(f"Incomplete run: {len(keyed)}/{len(expected)} unique model-image pairs")

    sizes = {image: float(keyed[(image, MODELS[0])]["largest_gt_fraction"]) for image in images}
    sorted_sizes = sorted(sizes.values())
    q1 = sorted_sizes[len(sorted_sizes) // 4]
    q3 = sorted_sizes[3 * len(sorted_sizes) // 4]
    groups = {
        "all": set(images),
        "smallest_quartile": {n for n, v in sizes.items() if v <= q1},
        "largest_quartile": {n for n, v in sizes.items() if v >= q3},
    }
    summary = []
    for group, members in groups.items():
        for model in MODELS:
            subset = [keyed[(image, model)] for image in sorted(members)]
            row = {"group": group, "model": model, "images": len(subset)}
            for field in ("hit_025", "aligned30_025", "aligned50_025",
                          "hit_050", "aligned30_050", "aligned50_050"):
                row[field] = sum(int(item[field]) for item in subset)
            row["median_latency_ms"] = round(statistics.median(float(item["latency_ms"]) for item in subset))
            summary.append(row)
    summary_path = OUT / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)

    report = [
        "# Flame detection on positive Roboflow images",
        "",
        f"Source: [Fire Data Annotations v5](https://universe.roboflow.com/fire-detection/fire-data-annotations/dataset/5). "
        f"The export has {cohort['published_test_images']} positive test images. "
        f"Exclude {cohort['test_images_sharing_source_id_with_train']} with a source ID also in train, "
        f"then take one image per remaining source ID: **{len(images)} images**.",
        "",
        "All four flame-capable local checkpoints processed the same images. "
        "A *hit* means at least one flame detection at the specified output-score cutoff. "
        "*Aligned* means at least one flame detection overlaps a published flame box at the specified IoU. "
        "These are image-level positive recall measures, not precision, specificity, COCO mAP, or an incident-level estimate.",
        "",
        "| Model | Hit at 0.25 | Aligned IoU≥0.30 at 0.25 | Aligned IoU≥0.50 at 0.25 | Hit at 0.50 | Aligned IoU≥0.50 at 0.50 | Median CPU ms/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    titles = {
        "dfire_yolov8n": "D-Fire YOLOv8n",
        "dfine": "FireViewer D-FINE M",
        "yolo": "FireViewer YOLO11-M",
        "soul": "SoulPerforms YOLOv8n (fire only)",
    }
    for model in MODELS:
        row = next(r for r in summary if r["group"] == "all" and r["model"] == model)
        n = row["images"]
        report.append(f"| {titles[model]} | {row['hit_025']}/{n} | {row['aligned30_025']}/{n} | "
                      f"{row['aligned50_025']}/{n} | {row['hit_050']}/{n} | "
                      f"{row['aligned50_050']}/{n} | {row['median_latency_ms']} |")
    report += [
        "",
        f"The largest annotated flame box covers at most {q1:.2%} of the image in the smallest quartile "
        f"and at least {q3:.2%} in the largest quartile. Size-stratified counts are in `summary.csv`.",
        "",
        "The images are all flame-positive. The exported labels have been format-checked and a small sample was visually reviewed, "
        "but the entire cohort has not been hand-corrected. The source-ID rule catches explicit Mirror/Noise copies; it does not "
        "prove separate incidents or exclude every near duplicate or model-training overlap. Output scores are not calibrated "
        "across models. The run cannot measure false alarms or alert timing; use verified negatives and whole clips for those.",
        "",
        "`per_image.csv` preserves each model-image result, output score, box alignment and CPU latency. "
        "`cohort_images.txt` lists the exact evaluation images. Reproduce with `python benchmark_roboflow_flame_positive.py` "
        "and `python summarize_roboflow_flame_positive.py`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved {summary_path} and {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
