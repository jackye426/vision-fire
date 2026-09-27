"""Compare saved model combinations on the estimated waste-pit shared clock.

This is a one-event, offline policy check. Model boxes come from the prior 1 fps
run; regions were chosen after viewing the clip. Clock times are edited-video
playback seconds, not real camera elapsed time.
"""

from __future__ import annotations

import csv
from itertools import combinations, permutations

from analyze_waste_pit_multiangle import (
    ANCHOR_A,
    ANCHOR_B,
    LAST_COMMON_SECOND,
    MODELS,
    OUT,
    angle1_video_time,
    angle2_video_time,
    boxes,
    iou,
    load_rows,
)
from shared_temporal_waste_pit import MATCH_IOU, SYNC_MARGIN_S


RESULT = OUT.with_name("multimodel_multiangle_waste_pit.csv")
END_A = ANCHOR_A + LAST_COMMON_SECOND
END_B = int(angle2_video_time(END_A))
ANGLES = ("angle1", "angle2")
LABELS = ("flame", "smoke")


def source_range(angle: str) -> range:
    return (range(ANCHOR_A, END_A + 1) if angle == "angle1" else
            range(int(ANCHOR_B), END_B + 1))


def shared_time(angle: str, source_second: int) -> float:
    return (float(source_second) if angle == "angle1" else
            angle1_video_time(source_second))


def first_hit(data: dict, model: str, label: str, angle: str,
              focused: bool) -> tuple[float, int] | None:
    for second in source_range(angle):
        if boxes(data, model, second, label, angle, focused):
            return shared_time(angle, second), second
    return None


def first_repeat(data: dict, model: str, label: str, angle: str,
                 focused: bool) -> tuple[float, int] | None:
    for second in source_range(angle):
        if second == source_range(angle).start:
            continue
        current = boxes(data, model, second, label, angle, focused)
        previous = boxes(data, model, second - 1, label, angle, focused)
        if any(iou(a, b) >= MATCH_IOU for a in current for b in previous):
            return shared_time(angle, second), second
    return None


def first_same_image(data: dict, model_a: str, model_b: str, label: str,
                     angle: str, focused: bool) -> tuple[float, int] | None:
    for second in source_range(angle):
        a_boxes = boxes(data, model_a, second, label, angle, focused)
        b_boxes = boxes(data, model_b, second, label, angle, focused)
        if any(iou(a, b) >= MATCH_IOU for a in a_boxes for b in b_boxes):
            return shared_time(angle, second), second
    return None


def first_three_model_image(data: dict, label: str, angle: str,
                            focused: bool) -> tuple[float, int] | None:
    """Require three boxes at one location, using D-FINE as the common anchor."""
    for second in source_range(angle):
        dfine = boxes(data, "dfine", second, label, angle, focused)
        dfire = boxes(data, "dfire_yolov8n", second, label, angle, focused)
        yolo = boxes(data, "yolo", second, label, angle, focused)
        if any(iou(anchor, fire) >= MATCH_IOU and iou(anchor, other) >= MATCH_IOU
               for anchor in dfine for fire in dfire for other in yolo):
            return shared_time(angle, second), second
    return None


def first_cross_view(data: dict, model_a: str, model_b: str, label: str,
                     focused: bool, margin_s: float = SYNC_MARGIN_S
                     ) -> tuple[float, int, int, float] | None:
    """Model A in angle 1 and model B in angle 2, with no future lookahead."""
    a_times = [a for a in source_range("angle1")
               if boxes(data, model_a, a, label, "angle1", focused)]
    b_times = [b for b in source_range("angle2")
               if boxes(data, model_b, b, label, "angle2", focused)]
    matches = []
    for a in a_times:
        for b in b_times:
            b_shared = angle1_video_time(b)
            gap = abs(a - b_shared)
            if gap <= margin_s:
                matches.append((max(a, b_shared), a, b, gap))
    return min(matches, key=lambda item: (item[0], item[3])) if matches else None


def main() -> None:
    data = load_rows()
    rows = []
    for focused in (False, True):
        for label in LABELS:
            for model in MODELS:
                for angle in ANGLES:
                    for rule, result in (
                        ("first_hit", first_hit(data, model, label, angle, focused)),
                        ("same_view_repeat", first_repeat(data, model, label, angle, focused)),
                    ):
                        if result is not None:
                            rows.append(dict(filter="reviewed_zone" if focused else "whole_frame",
                                             class_name=label, rule=rule,
                                             model_angle1=model if angle == "angle1" else "",
                                             model_angle2=model if angle == "angle2" else "",
                                             shared_clock_s=result[0],
                                             angle1_source_s=result[1] if angle == "angle1" else "",
                                             angle2_source_s=result[1] if angle == "angle2" else "",
                                             pair_gap_s=""))
            for model_a, model_b in combinations(MODELS, 2):
                for angle in ANGLES:
                    result = first_same_image(data, model_a, model_b, label,
                                              angle, focused)
                    if result is not None:
                        rows.append(dict(filter="reviewed_zone" if focused else "whole_frame",
                                         class_name=label, rule="same_image_model_agreement",
                                         model_angle1=f"{model_a}+{model_b}" if angle == "angle1" else "",
                                         model_angle2=f"{model_a}+{model_b}" if angle == "angle2" else "",
                                         shared_clock_s=result[0],
                                         angle1_source_s=result[1] if angle == "angle1" else "",
                                         angle2_source_s=result[1] if angle == "angle2" else "",
                                         pair_gap_s=""))
            for angle in ANGLES:
                result = first_three_model_image(data, label, angle, focused)
                if result is not None:
                    rows.append(dict(filter="reviewed_zone" if focused else "whole_frame",
                                     class_name=label, rule="same_image_three_model_agreement",
                                     model_angle1="dfine+dfire_yolov8n+yolo" if angle == "angle1" else "",
                                     model_angle2="dfine+dfire_yolov8n+yolo" if angle == "angle2" else "",
                                     shared_clock_s=result[0],
                                     angle1_source_s=result[1] if angle == "angle1" else "",
                                     angle2_source_s=result[1] if angle == "angle2" else "",
                                     pair_gap_s=""))
            for model_a, model_b in (*((model, model) for model in MODELS),
                                     *permutations(MODELS, 2)):
                result = first_cross_view(data, model_a, model_b, label, focused)
                if result is not None:
                    rows.append(dict(filter="reviewed_zone" if focused else "whole_frame",
                                     class_name=label, rule="cross_view_match",
                                     model_angle1=model_a, model_angle2=model_b,
                                     shared_clock_s=result[0],
                                     angle1_source_s=result[1], angle2_source_s=result[2],
                                     pair_gap_s=result[3]))
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {RESULT}")
    for row in rows:
        if row["filter"] == "reviewed_zone" and row["class_name"] == "flame" and row["rule"] in (
                "same_image_model_agreement", "cross_view_match"):
            print(row)


if __name__ == "__main__":
    main()
