# Shared incident context: waste-pit prototype

**Correction, 27 September 2026:** The [two waste-pit views](MULTIANGLE_WASTE_PIT.md) are sections of one edited video and play at different speeds. The former +3 s "two-view corroboration" and +4 s "persistent in both views" were produced by a fixed-offset replay, not simultaneous camera evidence. Those stages and timing claims have been removed from the current [code](shared_temporal_waste_pit.py), [timeline](outputs/youtube/shared_temporal_waste_pit.csv), and [sample incident record](outputs/youtube/shared_temporal_waste_pit_incident.json). The [local video](outputs/youtube/5NAkyEmC0IU_shared_temporal_h264.mp4) now uses an approximate visual speed correction from the original clip's grabber drop (7 s ↔ 59 s) and water cannon (31 s ↔ 71 s), and labels it as **not camera-clock synchronization**.

## What is shared now

This is an **offline, rule-based incident record** over detections already produced at 1 fps. The raw per-frame boxes stay in `outputs/youtube/frame_results.csv`. For each excerpt, the script reads the camera/view ID, source playback second, model, class, score and box. It applies a manually chosen fire/plume region for each view, checks whether D-FINE's box overlaps one in the *previous frame of the same view*, and checks whether D-FINE and D-Fire overlap on the *same image*. Only boxes within one view are compared by pixel coordinates.

The record groups observations under a known `waste_pit` physical-zone ID. It retains which views, models and classes produced detections, first/last flame and smoke source-video seconds **per camera**, same-image model-agreement times, the first/last **display index**, its watch/provisional stage, and whether it has emitted a provisional review notification. Later smoke or flame observations update that record rather than opening another one in this scripted replay. The timeline CSV exposes each per-view condition and `cross_view_temporal_enabled=0`. The JSON marks its mode `approximate_visual_alignment_offline_case_study`. Its one notification is a property of this hard-coded, one-incident replay; it is **not a measured reduction** in duplicate alerts.

The record does not retain stable track IDs, calibrated confidence, motion histories, learned video features, or a live camera feed. The fire regions were chosen after viewing this clip. There is no trustworthy shared clock here, so evidence from the second view does **not** advance the alert stage or verify a hit in the first view.

## What this case actually shows

Within angle 2, D-FINE repeatedly marked static clutter as flame at source 56–57 s. The second model did not overlap that early false box. At source 58 s, a tiny flame is visible and D-FINE and D-Fire box it at nearly the same location (IoU about 0.95); the demonstration's same-image model-agreement rule advances from watch to provisional review there. D-FINE also has a same-view flame repeat at 59 s. Under the new visual map, angle 1 source 5 s and angle 2 source 58 s fall in approximately the same display position, and D-FINE finds flame in both. That is an **illustrative co-occurrence**, exposed in the timeline as `approx_visual_pair_both_dfine_flame_not_alert`; it does not advance the alert stage. These are **per-view playback times**, not real camera elapsed time. They show that a repeated box alone can preserve a false detection and that model agreement can provide an additional signal on this frame. They do not show that a two-model or two-camera gate is reliably earlier or more accurate in general.

The [FURG outside-clip check](check_furg_model_agreement.py) uses saved 1 fps boxes from other videos and tests this same-camera D-FINE/D-Fire overlap rule (IoU ≥0.20, score ≥0.25). Its [per-clip table](outputs/furg/model_agreement_check.csv) shows:

| FURG sampled frames | D-FINE flame box | D-FINE + D-Fire overlapping flame boxes |
| --- | ---: | ---: |
| 79 flame-labeled frames across three clips | 71 | 61 |
| 102 frames from two flame-free clips | 3 | 0 |

On those flame-free clips, ordinary two-frame D-FINE persistence also produced zero alerts, so the overlap gate has no proven false-alert advantage there. On the positive `house1` clip, first D-FINE flame hit was at 13.013 s, its first location repeat at 14.014 s, and first D-FINE/D-Fire overlap only at 19.019 s. Requiring model agreement for every provisional alert could therefore delay or miss real fires. These are small frame checks, not a measured alert rate or multi-camera experiment.

## How shared context should work with real feeds

Each camera should maintain its own short history of `{capture timestamp, camera ID, model, class, score, box, local track, physical zone}`. A fixed camera-to-zone map would let the system join evidence about the same physical area without trying to overlap boxes from different perspectives. One incident record could then hold each view's latest evidence, camera health, first credible smoke/flame time, alert stage, and notification history. Capture timestamps must be synchronized or their offsets and drift measured before a rolling cross-camera window is meaningful. Model outputs from the same frame should be treated as correlated evidence, while different cameras can supply a more independent view if both truly cover the same event.

A sensible experiment would compare three policies on the **same simultaneous incidents and ordinary clear camera-hours**: per-camera persistence, same-camera model agreement, and cross-camera zone corroboration with a single-camera fallback. Measure time from first visible smoke/flame to provisional review and confirmed alert, missed events, false-alert episodes per camera-hour or site-hour, and duplicate notifications. Include dust, steam, hot work, moving equipment, occluded cameras, and separate fires in adjacent zones. The present edited clip supports the design of that experiment but cannot answer its cross-view performance questions.
