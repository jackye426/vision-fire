"""Audit a Roboflow YOLOv8 ZIP and render a small labeled contact sheet.

Usage: python audit_roboflow.py dataset/roboflow/example.zip
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SPLITS = ("train", "valid", "test")
PALETTE = ("#ff5733", "#8b5cf6", "#00a0d2", "#26a269", "#e5a50a")


def source_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    return re.sub(r"(?:_jpg|-jpg|_png|-png)$", "", stem, flags=re.I).lower()


def clip_stem(stem: str) -> str | None:
    # Roboflow commonly names extracted frames: video_mp4-0035 or clip_frame_0014.
    match = re.match(r"(.+?)(?:[_-](?:mp4|mov|avi)[_-]?\d+|[_-]frame[_-]?\d+)$", stem)
    return match.group(1) if match else None


def parse_names(yaml_text: str) -> list[str]:
    match = re.search(r"(?m)^names:\s*\[(.+?)\]\s*$", yaml_text)
    if match:
        return [x.strip().strip("'\"") for x in match.group(1).split(",")]
    return []


def audit(zip_path: Path, out_dir: Path, dfire_zip: Path | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        yaml_name = next((n for n in names if n.endswith("data.yaml")), None)
        classes = parse_names(z.read(yaml_name).decode()) if yaml_name else []
        image_paths = [n for n in names if "/images/" in n and n.lower().endswith(IMAGE_EXTS)]
        label_paths = {n for n in names if "/labels/" in n and n.endswith(".txt")}
        counts = collections.defaultdict(collections.Counter)
        class_images = collections.defaultdict(lambda: collections.defaultdict(set))
        source_splits = collections.defaultdict(set)
        clip_splits = collections.defaultdict(set)
        records = []
        invalid = []
        for image_path in image_paths:
            split = image_path.split("/", 1)[0]
            if split not in SPLITS:
                continue
            label_path = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            stem = source_stem(image_path)
            source_splits[stem].add(split)
            clip = clip_stem(stem)
            if clip:
                clip_splits[clip].add(split)
            counts[split]["images"] += 1
            if label_path not in label_paths:
                counts[split]["missing_label_file"] += 1
                lines = []
            else:
                lines = z.read(label_path).decode("utf-8-sig", errors="replace").splitlines()
            boxes = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if len(parts) == 5:
                        xywh = coords
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        xs, ys = coords[0::2], coords[1::2]
                        xywh = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                        counts[split]["polygon_label_lines"] += 1
                    else:
                        raise ValueError("unrecognized YOLO label shape")
                    if cls < 0 or cls >= len(classes) or not all(0 <= q <= 1 for q in coords) or xywh[2] <= 0 or xywh[3] <= 0:
                        raise ValueError("invalid class, coordinate, or line length")
                    boxes.append((cls, xywh))
                    counts[split][f"boxes_{classes[cls]}"] += 1
                    class_images[split][classes[cls]].add(image_path)
                except (ValueError, IndexError):
                    invalid.append({"path": label_path, "line": line[:120]})
            if not boxes:
                counts[split]["empty_images"] += 1
                class_images[split]["empty"].add(image_path)
            records.append({"image": image_path, "stem": stem, "split": split, "boxes": boxes})
        dfire_matches = []
        if dfire_zip and dfire_zip.exists():
            with zipfile.ZipFile(dfire_zip) as dz:
                dfire_stems = {source_stem(n) for n in dz.namelist() if "/images/" in n and n.lower().endswith(IMAGE_EXTS)}
            dfire_matches = sorted(dfire_stems.intersection(source_splits))
        cross_source = {k: sorted(v) for k, v in source_splits.items() if len(v) > 1}
        cross_clip = {k: sorted(v) for k, v in clip_splits.items() if len(v) > 1}
        report = {
            "zip": str(zip_path),
            "classes": classes,
            "splits": {s: dict(counts[s]) for s in SPLITS},
            "class_image_counts": {s: {k: len(v) for k, v in class_images[s].items()} for s in SPLITS},
            "invalid_label_lines": len(invalid),
            "invalid_examples": invalid[:12],
            "same_source_stem_across_splits": len(cross_source),
            "same_source_stem_examples": dict(list(cross_source.items())[:15]),
            "same_inferred_clip_across_splits": len(cross_clip),
            "same_inferred_clip_examples": dict(list(cross_clip.items())[:15]),
            "matching_dfire_source_stems": len(dfire_matches),
            "matching_dfire_examples": dfire_matches[:30],
        }
        (out_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Deterministic small sample, favoring the published test split.
        rng = random.Random(41)
        sample = []
        for category in [*classes, "empty"]:
            pool = sorted(class_images["test"].get(category, []))
            if len(pool) < 3:
                pool += sorted(class_images["valid"].get(category, []))
            sample += [(category, p) for p in rng.sample(pool, min(3, len(pool)))]
        if sample:
            tile_w, tile_h = 390, 330
            sheet = Image.new("RGB", (tile_w * 3, tile_h * ((len(sample) + 2) // 3)), "white")
            draw = ImageDraw.Draw(sheet)
            record_by_path = {r["image"]: r for r in records}
            for i, (category, path) in enumerate(sample):
                rec = record_by_path[path]
                with Image.open(io.BytesIO(z.read(path))) as opened:
                    image = opened.convert("RGB")
                image.thumbnail((tile_w - 16, tile_h - 70))
                x0 = i % 3 * tile_w + 8
                y0 = i // 3 * tile_h + 46
                sheet.paste(image, (x0, y0))
                draw.text((x0, y0 - 42), f"{rec['split']} / {category}", fill="black")
                draw.text((x0, y0 - 23), Path(path).name[:50], fill="black")
                for cls, (cx, cy, bw, bh) in rec["boxes"]:
                    left = x0 + (cx - bw / 2) * image.width
                    top = y0 + (cy - bh / 2) * image.height
                    right = x0 + (cx + bw / 2) * image.width
                    bottom = y0 + (cy + bh / 2) * image.height
                    color = PALETTE[cls % len(PALETTE)]
                    draw.rectangle((left, top, right, bottom), outline=color, width=3)
                    draw.text((left, max(y0, top - 16)), classes[cls], fill=color)
            sheet.save(out_dir / "contact_sheet.jpg", quality=90)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--dfire-zip", type=Path, default=Path("dataset/D-Fire.zip"))
    args = parser.parse_args()
    output = args.zip_path.parent / args.zip_path.stem
    print(json.dumps(audit(args.zip_path, output, args.dfire_zip), indent=2))
