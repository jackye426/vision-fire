"""Prototype a shared incident record for the two waste-pit views.

This replays saved 1 fps detector boxes. All rules are illustrative and use
post-hoc camera regions, so the output is a case study rather than a benchmark.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import subprocess

import cv2
import numpy as np

from analyze_waste_pit_multiangle import (
    ANCHOR_A,
    ANCHOR_B,
    LAST_COMMON_SECOND,
    OUT,
    REGIONS,
    SCORE,
    SOURCE,
    SECOND_A,
    SECOND_B,
    THIRD_A,
    THIRD_B,
    angle2_video_time,
    boxes,
    draw_text,
    iou,
    load_rows,
)


TIMELINE = OUT.with_name("shared_temporal_waste_pit.csv")
INCIDENT = OUT.with_name("shared_temporal_waste_pit_incident.json")
VIDEO = OUT.with_name("5NAkyEmC0IU_shared_temporal_h264.mp4")
MATCH_IOU = 0.20
STAGES = ("clear", "watch", "provisional")


@dataclass
class IncidentRecord:
    incident_id: str = "waste_pit_1"
    physical_zone: str = "waste_pit"
    stage: str = "clear"
    first_evidence_s: int | None = None
    last_evidence_s: int | None = None
    operator_alerts_emitted: int = 0
    supporting_cameras: set[str] = field(default_factory=set)
    supporting_models: set[str] = field(default_factory=set)
    observed_classes: set[str] = field(default_factory=set)
    milestones: dict[str, int] = field(default_factory=dict)
    camera_observations: dict[str, dict[str, int]] = field(default_factory=dict)

    def update(self, common: int, flame: dict[str, bool], smoke: dict[str, bool],
               model_agree: dict[str, bool], watch: bool, operator: bool) -> bool:
        """Accumulate evidence and emit at most one provisional review alert."""
        if any(flame.values()) or any(smoke.values()):
            if self.first_evidence_s is None:
                self.first_evidence_s = common
            self.last_evidence_s = common
            self.supporting_cameras.update(
                angle for angle in flame if flame[angle] or smoke[angle])
            self.supporting_models.add("dfine")
            if any(flame.values()):
                self.observed_classes.add("flame")
            if any(smoke.values()):
                self.observed_classes.add("smoke")
                self.milestones.setdefault("first_smoke", common)
        if any(model_agree.values()):
            self.supporting_models.add("dfire_yolov8n")
        for angle in flame:
            source_second = time_for(angle, common)
            observation = self.camera_observations.setdefault(angle, {})
            for label, present in (("flame", flame[angle]),
                                   ("smoke", smoke[angle]),
                                   ("same_image_model_agreement", model_agree[angle])):
                if present:
                    observation.setdefault(f"first_{label}_source_s", source_second)
                    observation[f"last_{label}_source_s"] = source_second
        if watch:
            self.milestones.setdefault("watch", common)
        target = ("provisional" if operator else
                  "watch" if watch else self.stage)
        if STAGES.index(target) > STAGES.index(self.stage):
            self.stage = target
            self.milestones.setdefault(target, common)
        new_operator_alert = self.stage in STAGES[2:] and not self.operator_alerts_emitted
        if new_operator_alert:
            self.operator_alerts_emitted = 1
        return new_operator_alert

    def as_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "physical_zone": self.physical_zone,
            "visual_alignment_landmarks_source_s": [
                {"angle1": ANCHOR_A, "angle2": ANCHOR_B},
                {"angle1": SECOND_A, "angle2": SECOND_B},
                {"angle1": THIRD_A, "angle2": THIRD_B},
            ],
            "cross_view_temporal_enabled": False,
            "temporal_mode": "approximate_visual_alignment_offline_case_study",
            "stage": self.stage,
            "first_evidence_replay_s": self.first_evidence_s,
            "last_evidence_replay_s": self.last_evidence_s,
            "operator_alerts_emitted": self.operator_alerts_emitted,
            "cameras_with_detections": sorted(self.supporting_cameras),
            "camera_observations_source_s": self.camera_observations,
            "supporting_models": sorted(self.supporting_models),
            "observed_classes": sorted(self.observed_classes),
            "milestones_replay_s": self.milestones,
        }


def time_for(angle: str, common: int) -> int:
    return (ANCHOR_A + common if angle == "angle1" else
            int(angle2_video_time(ANCHOR_A + common)))


def hit(data: dict, model: str, label: str, angle: str,
        common: int, focused: bool = True) -> bool:
    return bool(boxes(data, model, time_for(angle, common), label, angle, focused))


def repeat_at(data: dict, model: str, label: str, angle: str,
              common: int, focused: bool = True) -> bool:
    if common < 1:
        return False
    source_second = time_for(angle, common)
    current = boxes(data, model, source_second, label, angle, focused)
    # Compare adjacent frames within this camera, not two display positions;
    # the warped second view can repeat a source frame at adjacent positions.
    previous = boxes(data, model, source_second - 1, label, angle, focused)
    return any(iou(a, b) >= MATCH_IOU for a in current for b in previous)


def model_agreement(data: dict, label: str, angle: str,
                    common: int, focused: bool = True) -> bool:
    """D-FINE and D-Fire have overlapping same-class boxes in one image."""
    dfine = boxes(data, "dfine", time_for(angle, common), label, angle, focused)
    dfire = boxes(data, "dfire_yolov8n", time_for(angle, common),
                  label, angle, focused)
    return any(iou(a, b) >= MATCH_IOU for a in dfine for b in dfire)


def first(rows: list[dict], field: str) -> int | None:
    return next((int(row["replay_s"]) for row in rows if int(row[field])), None)


def render_video(data: dict, rows: list[dict]) -> None:
    """Show two edited views together, with their synchronization caveat."""
    if not SOURCE.exists():
        print(f"Skipped local video: {SOURCE} is absent")
        return
    source = cv2.VideoCapture(str(SOURCE))
    if not source.isOpened():
        raise RuntimeError(f"Cannot open {SOURCE}")
    source_fps = source.get(cv2.CAP_PROP_FPS)
    temp = VIDEO.with_name(VIDEO.stem + "_temp.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"),
                             5, (1280, 470))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {temp}")
    try:
        for index in range(34 * 5):
            common = index / 5
            sampled = int(common)
            state = rows[sampled]
            canvas = np.full((470, 1280, 3), (22, 22, 22), np.uint8)
            color = {"clear": (190, 190, 190), "watch": (255, 220, 100),
                     "provisional": (255, 210, 80)}[state["incident_stage"]]
            draw_text(canvas, f"Waste-pit shared incident: {state['incident_stage'].replace('_', ' ').upper()}",
                      (12, 28), 0.72, color)
            draw_text(canvas, f"2-LANDMARK VISUAL WARP / NO CAMERA SYNC | display +{common:.1f}s | "
                      f"operator alerts {state['operator_alerts_total']}",
                      (12, 55), 0.52)
            if state["new_operator_alert"]:
                draw_text(canvas, "NEW PROVISIONAL REVIEW", (930, 28), 0.58,
                          (80, 230, 255))
            for angle, left in (("angle1", 0), ("angle2", 640)):
                t = (ANCHOR_A + common if angle == "angle1" else
                     angle2_video_time(ANCHOR_A + common))
                source.set(cv2.CAP_PROP_POS_FRAMES, round(t * source_fps))
                ok, frame = source.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {angle} at {t:.1f}s")
                second = time_for(angle, sampled)
                for model in ("dfire_yolov8n", "dfine"):
                    for item in data[(model, second)]:
                        if float(item["score"]) < SCORE:
                            continue
                        label, box = item["label"], item["box"]
                        area = REGIONS[(angle, label)]
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        focused = area[0] <= cx <= area[2] and area[1] <= cy <= area[3]
                        if model == "dfire_yolov8n" and not focused:
                            continue
                        box_color = ((55, 190, 255) if label == "flame" else
                                     (70, 230, 95)) if model == "dfine" else (255, 185, 70)
                        if not focused:
                            box_color = (150, 150, 150)
                        thickness = (6 if model == "dfire_yolov8n" else 2) if focused else 1
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                                      box_color, thickness)
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                canvas[80:440, left:left + 640] = frame
                draw_text(canvas, f"{angle}: source {t:.1f}s; predictions {second}s",
                          (left + 8, 463), 0.52)
            writer.write(canvas)
    finally:
        writer.release()
        source.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(temp), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-an", str(VIDEO)], check=True)
    temp.unlink()
    print(f"Saved {VIDEO}")


def main() -> None:
    data = load_rows()
    rows = []
    incident = IncidentRecord()
    for common in range(LAST_COMMON_SECOND + 1):
        flame_hit = {angle: hit(data, "dfine", "flame", angle, common)
                     for angle in ("angle1", "angle2")}
        flame_repeat = {angle: repeat_at(data, "dfine", "flame", angle, common)
                        for angle in ("angle1", "angle2")}
        same_frame_models = {angle: model_agreement(data, "flame", angle, common)
                             for angle in ("angle1", "angle2")}
        # A shared incident advances through tiers; later evidence updates this
        # single incident instead of opening another camera/class alert.
        watch = any(flame_hit.values())
        # Repetition alone can preserve a false static-object track. Keep it as
        # track history; an operator review needs same-camera model agreement.
        operator = any(same_frame_models.values())
        smoke = {angle: hit(data, "dfine", "smoke", angle, common)
                 for angle in ("angle1", "angle2")}
        raw_repeat_angle2 = repeat_at(data, "dfine", "flame", "angle2",
                                      common, focused=False)
        raw_model_agreement = any(model_agreement(data, "flame", angle, common,
                                                  focused=False)
                                  for angle in ("angle1", "angle2"))
        new_alert = incident.update(common, flame_hit, smoke, same_frame_models,
                                    watch, operator)
        rows.append({
            "replay_s": common,
            "angle1_source_s": time_for("angle1", common),
            "angle2_source_s": time_for("angle2", common),
            "angle2_visual_mapped_s": round(angle2_video_time(ANCHOR_A + common), 3),
            "dfine_flame_angle1": int(flame_hit["angle1"]),
            "dfine_flame_angle2": int(flame_hit["angle2"]),
            "approx_visual_pair_both_dfine_flame_not_alert": int(all(flame_hit.values())),
            "dfine_flame_repeat_angle1": int(flame_repeat["angle1"]),
            "dfine_flame_repeat_angle2": int(flame_repeat["angle2"]),
            "dfine_dfire_flame_agree_angle1": int(same_frame_models["angle1"]),
            "dfine_dfire_flame_agree_angle2": int(same_frame_models["angle2"]),
            "dfine_smoke_angle1": int(smoke["angle1"]),
            "dfine_smoke_angle2": int(smoke["angle2"]),
            "watch_condition": int(watch),
            "operator_condition": int(operator),
            "cross_view_temporal_enabled": 0,
            "smoke_support_condition": int(any(smoke.values())),
            "raw_angle2_dfine_repeat_baseline": int(raw_repeat_angle2),
            "raw_same_frame_model_agreement": int(raw_model_agreement),
            "incident_stage": incident.stage,
            "new_operator_alert": int(new_alert),
            "operator_alerts_total": incident.operator_alerts_emitted,
            "incident_cameras": "|".join(sorted(incident.supporting_cameras)),
            "incident_classes": "|".join(sorted(incident.observed_classes)),
        })
    with TIMELINE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with INCIDENT.open("w", encoding="utf-8") as handle:
        json.dump(incident.as_dict(), handle, indent=2)
        handle.write("\n")
    render_video(data, rows)
    print(f"Saved {TIMELINE}")
    print(f"Saved {INCIDENT}")
    for field in ("watch_condition", "operator_condition", "smoke_support_condition"):
        seen = first(rows, field)
        print(f"{field}: {'disabled or absent' if seen is None else f'replay +{seen}s'}")
    print(f"raw single-camera repeat baseline: +"
          f"{first(rows, 'raw_angle2_dfine_repeat_baseline')}s")
    print(f"raw same-frame model agreement: +"
          f"{first(rows, 'raw_same_frame_model_agreement')}s")
    for angle in ("angle1", "angle2"):
        print(f"D-FINE/D-Fire flame agreement {angle}: +"
              f"{first(rows, f'dfine_dfire_flame_agree_{angle}')}s")
    # Whole-frame pre-fire boxes are deliberately checked as a separate risk.
    for common in (0, 1):
        print(f"raw angle2 +{common}s: D-FINE flame="
              f"{int(hit(data, 'dfine', 'flame', 'angle2', common, False))}, "
              f"D-FINE/D-Fire overlap="
              f"{int(model_agreement(data, 'flame', 'angle2', common, False))}, "
              f"other-view D-FINE flame="
              f"{int(hit(data, 'dfine', 'flame', 'angle1', common, False))}")


if __name__ == "__main__":
    main()
