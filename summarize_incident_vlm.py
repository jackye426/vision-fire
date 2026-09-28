"""Summarize the fixed checkpoint replay without treating frames as incidents."""

from __future__ import annotations

import csv
from itertools import combinations
import json
import statistics

from incident_vlm_replay import CHECKS, OUT, SCENES, iou, load_scene


RESULTS = OUT / "decisions.jsonl"
SUMMARY = OUT / "comparison.csv"


def first_any_two_model_flame(scene: str) -> int | None:
    frames = {}
    for item in load_scene(scene):
        if item["label"] == "flame":
            frames.setdefault((item["clock"], item["camera"], item["source_s"]), []).append(item)
    hits = [clock for (clock, _camera, _source_s), items in frames.items()
            if any(a["model"] != b["model"] and iou(a["box"], b["box"]) >= 0.20
                   for a, b in combinations(items, 2))]
    return min(hits) if hits else None


def main() -> None:
    records = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines()]
    by_key = {(r["policy"], r["check_id"]): r for r in records}
    if len(by_key) != len(CHECKS) * 2:
        raise ValueError("Expected one box and one incident result per fixed checkpoint")
    rows = []
    for scene, spec in SCENES.items():
        checks = sorted((c for c in CHECKS if c["scene"] == scene),
                        key=lambda c: c["clock"])
        for policy in ("rule", "any_two_models", "box", "incident"):
            alerts = []
            target_errors = []
            times = []
            costs = []
            active = False
            if policy == "any_two_models":
                first_pair = first_any_two_model_flame(scene)
                if first_pair is not None:
                    alerts.append(first_pair)
                    active = True
            for check in checks:
                if policy == "any_two_models":
                    alerted = False
                elif policy == "rule":
                    alerted = check["clock"] == spec["rule_first_alert"]
                else:
                    record = by_key[(policy, check["id"])]
                    answer = record["answer"]
                    proposal = (answer["decision"] == "yes" if policy == "box" else
                                answer["action"] == "escalate")
                    alerted = proposal and not active
                    times.append(record["elapsed_ms"])
                    costs.append(record["usage"].get("cost", 0))
                if alerted:
                    active = True
                    alerts.append(check["clock"])
                    if check["review"] == "no":
                        target_errors.append(check["clock"])
            first = min(alerts) if alerts else None
            onset = spec["true_flame_start"]
            # One incident notification per scene; subsequent positive checks
            # update it rather than creating one false episode per frame.
            false_incident_episode = int(scene == "cooler" and bool(alerts) or
                                         onset is not None and first is not None and first < onset)
            if policy == "any_two_models" and false_incident_episode:
                target_errors.append(first)
            rows.append({
                "scene": scene, "policy": policy,
                "first_alert_playback_s": first if first is not None else "",
                "flame_onset_playback_s": onset if onset is not None else "",
                "delay_from_onset_playback_s": first - onset if first is not None and onset is not None
                and not false_incident_episode else "",
                "false_incident_episodes_in_replay_window": false_incident_episode,
                "wrong_target_escalations_in_replay_window": len(target_errors),
                "median_api_ms_selected_calls": (statistics.median(times) if times else ""),
                "api_cost_usd_selected_calls": (round(sum(costs), 7) if costs else ""),
                "selected_checkpoints": len(checks),
            })
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved {SUMMARY}")


if __name__ == "__main__":
    main()
