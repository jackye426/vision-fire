# Shared temporal context across the two waste-pit views

This is a rule-based replay of the [rough 2 s ↔ 56 s alignment](MULTIANGLE_WASTE_PIT.md), using the saved 1 fps boxes at score ≥0.25. It explores what a shared *incident record* could do before training a new video model. The [20-second synchronized replay](outputs/youtube/5NAkyEmC0IU_shared_temporal_h264.mp4) shows the incident stage changing over both camera views. The [replay code](shared_temporal_waste_pit.py) writes a [per-second evidence timeline](outputs/youtube/shared_temporal_waste_pit.csv) and a [sample incident record](outputs/youtube/shared_temporal_waste_pit_incident.json). All times below are seconds after the assumed alignment anchor in an edited video, **not real camera elapsed seconds**.

## What the shared layer remembers

A production context should accept `{camera, timestamp, model, class, score, box, local track, physical zone}` for each detection. Each camera needs its own location tracks because the same flame has different pixel coordinates in the two views. Its camera-to-zone map connects those tracks to a shared physical incident.

**What this prototype actually writes:** raw boxes remain in `outputs/youtube/frame_results.csv`; the script recomputes two-frame box overlap and recent cross-camera hits from them. The JSON incident record stores one `waste_pit` incident ID, the assumed time alignment, first/last evidence times, supporting cameras/models/classes, stage milestones and one emitted operator alert. The timeline CSV stores the per-second conditions and stage. It does **not** persist track IDs, learned embeddings or motion histories. Its fire areas were chosen **after** viewing this positive clip; a real site needs a camera-to-zone map fixed before evaluation.

The incident advances through four illustrative stages:

1. **Watch, internal:** one credible flame box in the pit zone.
2. **Provisional operator review:** a second model overlaps that box in the same image. A repeated box from only one model stays in the internal watch state because static distractors can persist.
3. **Two-view corroboration:** D-FINE finds pit-zone flame in both cameras within a rolling two-second window. Pixel-box overlap is only used *within* a camera; the zone and time connect the views.
4. **Persistent in both views:** each camera has its own two-frame flame track within five recent aligned seconds. Later smoke evidence updates the same incident instead of opening another fire alert.

Model scores are ranking signals, **not calibrated probabilities**. D-FINE and D-Fire can also make correlated errors on the same image, so their same-frame agreement is weaker evidence than a second camera.

| Aligned time | Evidence seen in this clip | Illustrative state |
| ---: | --- | --- |
| +0 to +1 s | D-FINE repeatedly boxes unrelated objects in angle 2. D-Fire does not overlap those boxes; angle 1 has no flame box. | No operator alert under the two-evidence rule |
| **+2 s** | D-FINE and D-Fire boxes overlap on the tiny visible flame in angle 2 (box IoU 0.95). | **Provisional operator review** |
| **+3 s** | D-FINE also detects flame in angle 1; angle 2 now has a local repeat. | **Two-view corroboration** |
| **+4 s** | D-FINE has repeated flame boxes in both views. | Persistent two-view incident |
| +6 s | D-FINE adds smoke evidence in angle 2; angle 1's first pit-zone smoke box follows at +11 s. | Update the existing incident |

The first pit-zone same-frame D-FINE/D-Fire flame agreement in angle 1 is +4 s. The [source frame at angle 2's 58 s](outputs/youtube/5NAkyEmC0IU_source58.jpg) lets a person check that the +2 s box is on a small flame. The [synchronized local video](outputs/youtube/5NAkyEmC0IU_aligned_dfine_h264.mp4) shows the two views together.

## What this trial suggests

| Replay policy | First operator-level signal | Visual check |
| --- | ---: | --- |
| Whole-frame D-FINE, two-frame repeat in angle 2 | +1 s | False box on static clutter |
| Whole-frame D-FINE + D-Fire box agreement, or D-FINE from both views | **+2 s** | Overlapping boxes on the small flame |
| Fire-area D-FINE from both views | +3 s | Flame visible in both views |
| Fire-area D-FINE persistent in both views | +4 s | Stronger but slower evidence |

The previous **fire-area** single-camera, two-frame D-FINE rule first alerts from angle 2 at **+3 s**. Allowing an overlapping D-FINE and D-Fire hit to request *provisional* review at **+2 s** gains one playback second in this example; **that gain comes from using two models on one camera**. The shared second camera corroborates at **+3 s**, so two-view verification itself is no faster than the best single camera here. From angle 1's perspective, however, the second view corroborates its first +3 s hit immediately, rather than waiting for angle 1's local repeat at +4 s. Requiring both persistent cameras waits until **+4 s**.

The shared record also collapses four separate D-FINE camera/class repeat signals—flame from each angle and smoke from each angle—into **one operator alert** with later evidence updates. That reduces duplicate notifications, which is different from reducing false alarms. In the *whole-frame* boxes before the post-hoc fire-area filter, D-FINE's early angle-2 false flame track at +0/+1 would satisfy the **single-camera repeat baseline at +1 s**. It has neither D-Fire overlap nor angle-1 flame support, so the stricter shared rule would withhold an operator alert. The replay also uses the manually selected fire area, so it does not isolate how much each filter contributes. One edited positive clip has only a very short pre-fire period and cannot estimate false alerts per camera-hour.

The record retains its strongest stage during this one incident. Its stage transitions are flame-triggered, with smoke added as supporting evidence; smoke-only events need their own path. A stricter cross-evidence requirement can also miss a genuine one-camera fire if the other camera and model both fail, so a production design needs a separately tested single-camera fallback. A real temporal layer also needs an expiry/cooldown rule and a way to keep two simultaneous fires in different zones as separate incidents; this replay does not test those behaviors.

## Small outside-clip check of the provisional rule

The [FURG check](check_furg_model_agreement.py) reuses the existing saved 1 fps flame boxes. It compares a D-FINE flame hit with **same-frame, overlapping D-FINE and D-Fire boxes** (IoU ≥0.20) at score ≥0.25. This checks only the same-camera *model-agreement* tier; FURG does not supply synchronized pairs of camera views. The [per-clip result table](outputs/furg/model_agreement_check.csv) has the exact counts.

| FURG sampled frames | D-FINE flame box | D-FINE + D-Fire overlap |
| --- | ---: | ---: |
| 79 frames with a flame label, across three clips | 71 | 61 |
| 102 frames from two flame-free clips | 3 | 0 |

The gate removed three isolated D-FINE hits on the flame-free clips, but ordinary two-frame D-FINE persistence also produced **zero alerts** there. The positive-frame gate loses ten D-FINE-hit frames. In `house1`, first D-FINE flame hit was at clip 13.013 s, first location repeat at 14.014 s, and first D-FINE/D-Fire overlap at 19.019 s. Same-frame model agreement therefore cannot be the mandatory route to review: it can be fast in the waste-pit example and late on a different fire. These are small image-level transfer checks, not localization-confirmed agreement or a false-alert rate per camera-hour.

The next policy to test is **tiered**: preserve a single-camera watch silently; allow immediate provisional review when either another model agrees or a second camera sees the same physical zone; offer a separate lower-confidence fallback for a persistent, changing single-camera flame/smoke track when the other sources miss it; and update one incident as new views or classes arrive. The fallback's persistence and motion thresholds must be chosen on development footage, then checked on unseen clear camera-hours. An offline or occluded camera must not silently veto a real incident.

For an operational test, keep the camera clock and zone map fixed, replay truly simultaneous footage, and compare three alert policies on the **same incidents and ordinary clear camera-hours**: single-camera persistence, same-camera model agreement, and the shared incident stages above. Record first provisional review, first two-view confirmation, missed incidents, duplicate notifications, and false alert episodes per camera-hour. Include dust, steam, headlights, hot work, and non-overlapping fires so the shared layer can be tested for wrongly merging events.
