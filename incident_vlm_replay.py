"""Replay three-detector incident histories against two OpenRouter VLM policies.

Fixed checkpoints are a causal case study, not an automated full-clip benchmark.
Both VLM policies see the same current images. Only the incident policy receives
the three-model history, overlap/repeat/cross-view summaries, and prior actions.
No labels, future detections, or outcome times enter either VLM prompt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2

from analyze_waste_pit_multiangle import REGIONS
from openrouter_vlm_pilot import (DEFAULT_MODEL, ROOT, call_api,
                                  image_part, key_from_environment,
                                  region_crop, save_jpeg)


OUT = ROOT / "outputs" / "incident_vlm"
MODELS = ("dfire_yolov8n", "dfine", "yolo")
SCORE = 0.25
IOU = 0.20
FURG = ROOT / "outputs" / "furg" / "frame_results.csv"
YOUTUBE = ROOT / "outputs" / "youtube" / "frame_results.csv"

# Checkpoint IDs and target boxes were frozen before these incident API calls.
# Waste-pit angle 2 is mapped by two user-identified landmarks:
# shared angle-1 playback second = 2 * (angle-2 source second - 55.5).
# The mapping is approximate and does not create true capture timestamps.
CHECKS = (
    dict(id="pit_1_static", scene="pit", clock=1, camera="angle2", source_s=56,
         label="flame", box=[481, 133, 537, 150], review="no"),
    dict(id="pit_3_static_repeat", scene="pit", clock=3, camera="angle2", source_s=57,
         label="flame", box=[481, 133, 537, 150], review="no"),
    dict(id="pit_5_flame", scene="pit", clock=5, camera="angle2", source_s=58,
         label="flame", box=[813, 409, 853, 450], review="yes"),
    dict(id="pit_12_early_smoke", scene="pit", clock=12, camera="angle1", source_s=12,
         label="smoke", box=[588, 504, 641, 597], review="uncertain"),
    dict(id="pit_13_smoke", scene="pit", clock=13, camera="angle1", source_s=13,
         label="smoke", box=[560, 441, 661, 607], review="yes"),
    dict(id="house_13_first_flame", scene="house", clock=13, camera="house_cam",
         source_s=13, label="flame", box=[230, 174, 263, 221], review="yes"),
    dict(id="house_14_repeat", scene="house", clock=14, camera="house_cam",
         source_s=14, label="flame", box=[209, 148, 252, 223], review="yes"),
    dict(id="house_19_agreement", scene="house", clock=19, camera="house_cam",
         source_s=19, label="flame", box=[170, 29, 266, 227], review="yes"),
    dict(id="cooler_6_false", scene="cooler", clock=6, camera="cooler_cam",
         source_s=6, label="flame", box=[380, 274, 411, 329], review="no"),
    dict(id="cooler_7_false", scene="cooler", clock=7, camera="cooler_cam",
         source_s=7, label="flame", box=[417, 267, 450, 327], review="no"),
    dict(id="cooler_15_false", scene="cooler", clock=15, camera="cooler_cam",
         source_s=15, label="flame", box=[622, 319, 655, 344], review="no"),
)

SCENES = {
    "pit": {"clip": "5NAkyEmC0IU", "source": "youtube", "first": 0,
            "last": 13, "true_flame_start": 2, "rule_first_alert": 5,
            "clock_note": "Estimated shared playback clock; angle 2 has a visual time warp.",
            "camera_zone": {"angle1": "waste_pit", "angle2": "waste_pit"}},
    "house": {"clip": "house1", "source": "furg", "first": 0,
              "last": 19, "true_flame_start": 5, "rule_first_alert": 19,
              "clock_note": "Single source playback clock, approximately 1 fps.",
              "camera_zone": {"house_cam": "house_scene"}},
    "cooler": {"clip": "coolerbot", "source": "furg", "first": 0,
               "last": 15, "true_flame_start": None, "rule_first_alert": None,
               "clock_note": "Single source playback clock, approximately 1 fps.",
               "camera_zone": {"cooler_cam": "robot_scene"}},
}

INCIDENT_SCHEMA = {
    "type": "object", "properties": {
        "action": {"type": "string", "enum": ["watch", "request_other_view",
                                                "escalate", "update_incident"]},
        "hazard": {"type": "string", "enum": ["flame", "smoke", "none", "uncertain"]},
        "evidence": {"type": "string", "description": "At most 30 words of concrete evidence"},
        "concern": {"type": "string", "description": "Main uncertainty or confuser, or empty string"},
    }, "required": ["action", "hazard", "evidence", "concern"],
    "additionalProperties": False,
}
BOX_SCHEMA = {
    "type": "object", "properties": {
        "decision": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "evidence": {"type": "string", "description": "At most 25 words of visible evidence"},
    }, "required": ["decision", "evidence"], "additionalProperties": False,
}


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return area / max(aa + bb - area, 1)


def camera_time(scene: str, source_s: int) -> tuple[str, int] | None:
    if scene == "pit":
        if 0 <= source_s <= 13:
            return "angle1", source_s
        if 56 <= source_s <= 62:
            return "angle2", 2 * source_s - 111
        return None
    return (("house_cam" if scene == "house" else "cooler_cam"), source_s)


def load_scene(scene: str) -> list[dict]:
    spec = SCENES[scene]
    path = YOUTUBE if spec["source"] == "youtube" else FURG
    loaded: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != spec["clip"] or row["model"] not in MODELS:
                continue
            source_s = round(float(row["time_s"]))
            mapped = camera_time(scene, source_s)
            if mapped is None:
                continue
            camera, clock = mapped
            if not spec["first"] <= clock <= spec["last"]:
                continue
            raw = json.loads(row["detections"] if scene == "pit" else row["predictions"])
            for index, item in enumerate(raw):
                if float(item["score"]) < SCORE or item["label"] not in ("flame", "smoke"):
                    continue
                loaded.append({"id": f"{camera}:{source_s}:{row['model']}:{index}",
                               "clock": clock, "camera": camera, "source_s": source_s,
                               "model": row["model"], "label": item["label"],
                               "score": round(float(item["score"]), 3),
                               "box": item["box"]})
    loaded.sort(key=lambda item: (item["clock"], item["camera"], item["model"],
                                  -item["score"]))
    return loaded


def check_target(check: dict, observations: list[dict]) -> None:
    if not any(x["camera"] == check["camera"] and x["source_s"] == check["source_s"]
               and x["label"] == check["label"] and x["box"] == check["box"]
               for x in observations):
        raise ValueError(f"Target box not found in saved predictions: {check['id']}")


def frame_at(scene: str, source_s: int):
    spec = SCENES[scene]
    media_dir = ROOT / "dataset" / ("youtube" if spec["source"] == "youtube" else "furg")
    path = media_dir / f"{spec['clip']}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(source_s * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {path.name} at {source_s}s")
    return frame


def prepare_images(check: dict) -> dict[str, Path]:
    prefix = OUT / "inputs" / check["id"]
    current = frame_at(check["scene"], check["source_s"])
    paths = {"current_full": prefix.with_name(prefix.name + "_full.jpg"),
             "target_crop": prefix.with_name(prefix.name + "_crop.jpg")}
    save_jpeg(current, paths["current_full"])
    save_jpeg(region_crop(current, check["box"]), paths["target_crop"])
    if check["scene"] == "pit":
        # Equal shared-clock source frame in the other edited view when present;
        # at 12 s, angle-2 source 61 is the most recent available (clock 11).
        other_s = (round((check["clock"] + 111) / 2)
                   if check["camera"] == "angle1" else check["clock"])
        if check["camera"] == "angle1":
            other_s = min(other_s, (check["clock"] + 111) // 2)
        paths["other_view"] = prefix.with_name(prefix.name + "_other.jpg")
        save_jpeg(frame_at("pit", other_s), paths["other_view"])
    return paths


def target_summary(check: dict, history: list[dict]) -> dict:
    current = [o for o in history if o["camera"] == check["camera"]
               and o["source_s"] == check["source_s"] and o["label"] == check["label"]]
    supporting = [o for o in current if iou(o["box"], check["box"]) >= IOU]
    prior = [o for o in history if o["camera"] == check["camera"]
             and o["source_s"] < check["source_s"] and o["label"] == check["label"]
             and o["model"] in {s["model"] for s in supporting}
             and iou(o["box"], check["box"]) >= IOU]
    other = [o for o in history if o["camera"] != check["camera"]
             and o["label"] == check["label"]
             and abs(o["clock"] - check["clock"]) <= 1]
    def in_pit_zone(camera: str, label: str, box: list[int]) -> bool:
        area = REGIONS[(camera, label)]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return area[0] <= cx <= area[2] and area[1] <= cy <= area[3]

    zone_links = (other if check["scene"] == "pit" and
                  in_pit_zone(check["camera"], check["label"], check["box"])
                  else [])
    zone_links = [o for o in zone_links if in_pit_zone(o["camera"], o["label"], o["box"])]
    return {
        "current_box": check["box"], "class": check["label"],
        "same_location_models": sorted({o["model"] for o in supporting}),
        "models_without_same_location_box": sorted(set(MODELS) -
                                                   {o["model"] for o in supporting}),
        "prior_same_location_sample_times": sorted({o["source_s"] for o in prior}),
        "other_camera_same_class_raw_observations": [
            {"camera": o["camera"], "clock": o["clock"], "model": o["model"],
             "box": o["box"], "score": o["score"]} for o in other[:12]],
        "same_physical_zone_cross_view_links": [
            {"camera": o["camera"], "clock": o["clock"], "model": o["model"],
             "box": o["box"]} for o in zone_links[:12]],
    }


def history_packet(check: dict, observations: list[dict], prior_actions: list[dict]) -> dict:
    history = [o for o in observations if o["clock"] <= check["clock"]]
    recent = [o for o in history if o["clock"] >= check["clock"] - 4]
    # Compact repeated duplicate boxes while retaining at least one per
    # camera/model/class/source second. All raw observations remain in memory.
    recent_best = {}
    for o in recent:
        key = (o["camera"], o["source_s"], o["model"], o["label"])
        if key not in recent_best or o["score"] > recent_best[key]["score"]:
            recent_best[key] = o
    counts = defaultdict(lambda: {"frames": set(), "first": None, "last": None})
    for o in history:
        key = (o["camera"], o["model"], o["label"])
        item = counts[key]
        item["frames"].add(o["source_s"])
        item["first"] = o["clock"] if item["first"] is None else min(item["first"], o["clock"])
        item["last"] = o["clock"] if item["last"] is None else max(item["last"], o["clock"])
    cumulative = [{"camera": c, "model": m, "class": label,
                   "sampled_frames_with_hit": len(value["frames"]),
                   "first_clock": value["first"], "last_clock": value["last"]}
                  for (c, m, label), value in sorted(counts.items())]
    return {
        "shared_incident_id": check["scene"],
        "clock": check["clock"],
        "clock_note": SCENES[check["scene"]]["clock_note"],
        "camera_to_physical_zone": SCENES[check["scene"]]["camera_zone"],
        "zone_mapping_note": ("Waste-pit regions were selected after seeing this clip; "
                              "do not treat raw same-class boxes across cameras as "
                              "the same physical location." if check["scene"] == "pit" else
                              "One camera; no cross-view link available."),
        "current_camera": check["camera"],
        "current_source_second": check["source_s"],
        "current_target": target_summary(check, history),
        "cumulative_detection_summary": cumulative,
        "recent_detections": sorted(recent_best.values(),
                                    key=lambda o: (o["clock"], o["camera"], o["model"])),
        "earlier_policy_actions": prior_actions,
    }


def request_body(check: dict, policy: str, packet: dict,
                 paths: dict[str, Path], model: str, already_alerted: bool) -> dict:
    description = ("Images: current full frame; marked crop of the current proposal; "
                   "then the other camera's latest available frame if supplied. "
                   "The cyan outline marks only a detector proposal, not ground truth. "
                   "Do not use an unmarked fire elsewhere to validate that box. ")
    if policy == "incident":
        system = ("You manage one evolving camera incident. Use ONLY evidence in "
                  "the supplied history and images available at this checkpoint. "
                  "Detector scores are uncalibrated. Same-model repeat can preserve "
                  "a static false object; agreement can share an error. Across-camera "
                  "boxes have different pixel coordinates. Return watch, "
                  "request_other_view, escalate for operator review, or update_incident. "
                  "Escalate only for visually credible flame or smoke or convincing "
                  "independent evidence. If an operator alert already exists, use "
                  "update_incident for new supporting evidence instead of escalating again. "
                  "Be explicit about uncertainty. No future frames are available.")
        message = (description + f"Alert already issued: {already_alerted}. "
                   "Causal evidence history follows:\n" + json.dumps(packet,
                   separators=(",", ":")))
        schema, name = INCIDENT_SCHEMA, "incident_decision"
    else:
        system = ("Judge only whether the requested phenomenon is visibly present "
                  "inside the cyan box in the current crop. A box is a detector "
                  "proposal, not ground truth. Flame is visible combustion, not "
                  "lights or paint. Smoke is an airborne plume, not a solid surface. "
                  "If pixels are insufficient, answer uncertain.")
        message = description + f"Is {check['label']} visible inside the cyan box?"
        schema, name = BOX_SCHEMA, "box_decision"
    content = [{"type": "text", "text": message},
               image_part(paths["current_full"]), image_part(paths["target_crop"])]
    if "other_view" in paths:
        content.append(image_part(paths["other_view"]))
    return {"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": name, "strict": True, "schema": schema}},
        "provider": {"require_parameters": True},
        "temperature": 0, "max_tokens": 220, "stream": False}


def read_results(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {(r["policy"], r["check_id"]): r for r in records}


def run_policy(policy: str, checks: list[dict], scenes: dict[str, list[dict]],
               model: str, key: str | None, results: dict, path: Path) -> None:
    prior_actions: dict[str, list[dict]] = defaultdict(list)
    alerted: dict[str, bool] = defaultdict(bool)
    for check in checks:
        check_target(check, scenes[check["scene"]])
        paths = prepare_images(check)
        packet = history_packet(check, scenes[check["scene"]], prior_actions[check["scene"]])
        packet_path = OUT / "packets" / f"{policy}_{check['id']}.json"
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
        if key is None:
            print(f"Prepared {policy} {check['id']}", flush=True)
            continue
        record_key = policy, check["id"]
        if record_key in results:
            record = results[record_key]
        else:
            body = request_body(check, policy, packet, paths, model,
                                alerted[check["scene"]])
            response, elapsed_ms = call_api(body, key)
            answer = json.loads(response["choices"][0]["message"]["content"])
            record = {
                "policy": policy, "check_id": check["id"], "scene": check["scene"],
                "clock": check["clock"], "camera": check["camera"],
                "source_s": check["source_s"], "target": check["label"],
                "review": check["review"], "model": model,
                "answer": answer, "elapsed_ms": elapsed_ms,
                "usage": response.get("usage", {}),
                "input_sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for name, p in paths.items()},
                "history_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        answer = record["answer"]
        if policy == "incident":
            action = answer["action"]
            if action == "escalate":
                alerted[check["scene"]] = True
            decision = f"{action}/{answer['hazard']}"
        else:
            action = "escalate" if answer["decision"] == "yes" and not alerted[check["scene"]] else (
                "update_incident" if answer["decision"] == "yes" else "watch")
            if action == "escalate":
                alerted[check["scene"]] = True
            decision = answer["decision"]
        prior_actions[check["scene"]].append({"clock": check["clock"],
                                              "action": action, "target": check["label"]})
        print(f"{policy} {check['id']}: {decision} "
              f"({record['elapsed_ms']} ms)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--policy", choices=("incident", "box", "both"), default="both")
    args = parser.parse_args()
    scenes = {scene: load_scene(scene) for scene in SCENES}
    checks = list(CHECKS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "decisions.jsonl"
    results = read_results(path)
    key = key_from_environment() if args.run else None
    policies = ("box", "incident") if args.policy == "both" else (args.policy,)
    for policy in policies:
        run_policy(policy, checks, scenes, args.model, key, results, path)
    if not args.run:
        print("Prepared causal packets and images. Add --run to call OpenRouter.")


if __name__ == "__main__":
    main()
