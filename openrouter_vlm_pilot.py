"""Small, causal OpenRouter VLM check on saved fire/smoke detector boxes.

The VLM sees the original frame and an annotated crop of ONE proposed box.
Labels below are a small visual case review, not a benchmark ground truth set.
No model name, detector score, answer, or future frame is sent in the prompt.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2


ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "dataset" / "youtube"
DETECTIONS = ROOT / "outputs" / "youtube" / "frame_results.csv"
OUT = ROOT / "outputs" / "vlm_validator"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
API = "https://openrouter.ai/api/v1/chat/completions"

# Each target box comes from the saved 1 fps detector sweep. The repeated boxes
# are kept for qualitative context but must not be counted as independent events.
CASES = (
    dict(id="pit_b56_false", clip="5NAkyEmC0IU", second=56, detector="dfine",
         target="flame", box=[481, 133, 537, 150], review="no",
         group="pit_angle2_static_object"),
    dict(id="pit_b57_false_repeat", clip="5NAkyEmC0IU", second=57,
         detector="dfine", target="flame", box=[481, 133, 537, 150],
         review="no", group="pit_angle2_static_object"),
    dict(id="pit_a5_tiny_flame", clip="5NAkyEmC0IU", second=5,
         detector="dfine", target="flame", box=[622, 614, 660, 654],
         review="yes", group="pit_fire"),
    dict(id="pit_b58_tiny_flame", clip="5NAkyEmC0IU", second=58,
         detector="dfine", target="flame", box=[813, 409, 853, 450],
         review="yes", group="pit_fire"),
    dict(id="pit_a12_early_smoke", clip="5NAkyEmC0IU", second=12,
         detector="yolo", target="smoke", box=[588, 504, 641, 597],
         review="uncertain", group="pit_smoke"),
    dict(id="pit_a13_smoke", clip="5NAkyEmC0IU", second=13,
         detector="yolo", target="smoke", box=[560, 441, 661, 607],
         review="yes", group="pit_smoke"),
    dict(id="battery_7_excavator_false", clip="CI6YpclCYA4", second=7,
         detector="dfire_yolov8n", target="flame", box=[548, 233, 671, 348],
         review="no", group="battery_excavator"),
    dict(id="battery_10_flame", clip="CI6YpclCYA4", second=10,
         detector="yolo", target="flame", box=[252, 516, 287, 560],
         review="yes", group="battery_fire"),
    dict(id="recycling_10_smoke", clip="WsUjSE-ibKo", second=10,
         detector="dfire_yolov8n", target="smoke", box=[549, 5, 952, 270],
         review="yes", group="recycling_smoke"),
)
TEMPORAL_CASES = {"pit_b56_false", "pit_b58_tiny_flame", "pit_a12_early_smoke"}

SYSTEM = ("You are checking one camera detector proposal. Judge only the marked "
          "region in the current frame. Be conservative. Flame means visible "
          "combustion, not an orange machine, lamp, reflection, or painted object. "
          "Smoke means a visible airborne plume, not a solid surface, steam, "
          "dust, or compression artifact. If pixels do not resolve the target, "
          "answer uncertain. The full frame supplies context. The outlined crop "
          "identifies the proposed region; its colored outline is an annotation.")
SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["yes", "no", "uncertain"],
                     "description": "Whether the requested phenomenon is visible inside the marked region"},
        "visual_evidence": {"type": "string", "description": "Brief concrete visible evidence, at most 20 words"},
        "possible_confuser": {"type": "string", "description": "Visible alternative or empty string"},
    },
    "required": ["decision", "visual_evidence", "possible_confuser"],
    "additionalProperties": False,
}


def key_from_environment() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    local = ROOT / ".env.local"
    if local.exists():
        for line in local.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("Set OPENROUTER_API_KEY in the environment or ignored .env.local")


def verify_saved_box(case: dict) -> None:
    with DETECTIONS.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if row["clip"] == case["clip"] and row["model"] == case["detector"]
                   and int(row["time_s"]) == case["second"]]
    if len(matches) != 1 or not any(
        d["label"] == case["target"] and d["box"] == case["box"]
        and float(d["score"]) >= 0.25
        for d in json.loads(matches[0]["detections"])
    ):
        raise ValueError(f"Saved detector box changed or missing: {case['id']}")


def source_frame(case: dict, second: int):
    path = MEDIA / f"{case['clip']}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(second * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path.name} at {second}s")
    return frame


def region_crop(frame, box: list[int]):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    side = min(max(256, 4 * max(x2 - x1, y2 - y1)), max(width, height))
    left = max(0, min(width - side, mid_x - side // 2))
    top = max(0, min(height - side, mid_y - side // 2))
    crop = frame[top:top + side, left:left + side].copy()
    cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top),
                  (255, 255, 0), 2)  # cyan in BGR; annotation only
    return crop


def save_jpeg(frame, path: Path) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError(f"JPEG encode failed: {path}")
    data = encoded.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def prepare_case(case: dict) -> dict[str, Path]:
    verify_saved_box(case)
    frame = source_frame(case, case["second"])
    prefix = OUT / "inputs" / case["id"]
    paths = {"full": prefix.with_name(prefix.name + "_full.jpg"),
             "crop": prefix.with_name(prefix.name + "_target.jpg")}
    save_jpeg(frame, paths["full"])
    save_jpeg(region_crop(frame, case["box"]), paths["crop"])
    if case["id"] in TEMPORAL_CASES:
        paths["prior"] = prefix.with_name(prefix.name + "_prior.jpg")
        save_jpeg(region_crop(source_frame(case, case["second"] - 1), case["box"]),
                  paths["prior"])
    return paths


def image_part(path: Path) -> dict:
    data = path.read_bytes()
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}}


def request_body(case: dict, paths: dict[str, Path], model: str,
                 temporal: bool) -> dict:
    question = (f"Is {case['target']} visibly present INSIDE the cyan outline "
                "in the current target crop? Reply yes, no, or uncertain. "
                "Do not judge another location in the scene. Images: current "
                "full frame, then current target crop.")
    content = [{"type": "text", "text": question}, image_part(paths["full"]),
               image_part(paths["crop"])]
    if temporal:
        content[0]["text"] += (" The final image is the SAME region one sampled "
                               "second earlier; use it only as supporting context. "
                               "Judge the current image, not a future outcome.")
        content.append(image_part(paths["prior"]))
    return {"model": model, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "fire_smoke_box_review", "strict": True, "schema": SCHEMA}},
        "provider": {"require_parameters": True},
        "temperature": 0, "max_tokens": 200, "stream": False}


def call_api(body: dict, key: str) -> tuple[dict, float]:
    req = urllib.request.Request(API, data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not log request headers or the API key.
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from None
    return result, round((time.perf_counter() - started) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Send paid API requests")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--case", action="append", help="Limit to a case ID; repeatable")
    parser.add_argument("--temporal", action="store_true",
                        help="Add prior-frame crop for three selected cases")
    args = parser.parse_args()
    selected = [case for case in CASES if not args.case or case["id"] in args.case]
    if args.case and len(selected) != len(set(args.case)):
        raise ValueError("Unknown or repeated case ID")
    if args.temporal:
        selected = [case for case in selected if case["id"] in TEMPORAL_CASES]
    key = key_from_environment() if args.run else ""
    OUT.mkdir(parents=True, exist_ok=True)
    result_file = OUT / ("temporal_results.jsonl" if args.temporal else "single_results.jsonl")
    completed = set()
    if result_file.exists():
        for line in result_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            completed.add((record["case_id"], record["model"]))
    for case in selected:
        paths = prepare_case(case)
        print(f"Prepared {case['id']}: {paths['full'].name}, {paths['crop'].name}",
              flush=True)
        if not args.run or (case["id"], args.model) in completed:
            continue
        body = request_body(case, paths, args.model, args.temporal)
        result, elapsed_ms = call_api(body, key)
        answer = json.loads(result["choices"][0]["message"]["content"])
        if answer["decision"] not in ("yes", "no", "uncertain"):
            raise ValueError(f"Unexpected answer for {case['id']}")
        record = {"case_id": case["id"], "clip": case["clip"],
                  "source_second": case["second"], "target": case["target"],
                  "box": case["box"], "review": case["review"],
                  "event_group": case["group"], "model": args.model,
                  "temporal": args.temporal, "elapsed_ms": elapsed_ms,
                  "answer": answer, "usage": result.get("usage", {}),
                  "provider": result.get("provider"),
                  "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for name, path in paths.items()}}
        with result_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"{case['id']}: {answer['decision']} ({elapsed_ms} ms)", flush=True)
    if not args.run:
        print("Prepared images only. Add OPENROUTER_API_KEY and pass --run to query.")


if __name__ == "__main__":
    main()
