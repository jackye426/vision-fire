# Shared playback clock and incident context: waste-pit trial

## Clock and matching rule

The two sections of the [edited waste-pit video](MULTIANGLE_WASTE_PIT.md) play at different speeds. The user identified the same grabber drop at angle 1 **7 s** / angle 2 **59 s** and water-cannon start at **31 s** / **71 s** in the original clip. We use `angle 2 second = 55.5 + 0.5 × angle 1 second`, so angle 1 playback seconds are the shared reference clock. The earlier dozer estimate at angle 1 2 s maps to angle 2 56.5 s. This is an **estimated shared playback clock**, not a camera capture timestamp; no real elapsed detection delay can be calculated from it.

Accepting the user's estimate of at most about **±1 second of alignment error**, the [replay code](shared_temporal_waste_pit.py) pairs D-FINE boxes when they have the same class, fall in manually reviewed regions mapped to the same physical waste-pit zone, and their shared-clock times differ by at most one angle-1 playback second. Boxes are never matched by pixel coordinates across cameras. A pair is available only after **both** source frames have occurred on that clock. Angle 1 was sampled at 1 fps; angle 2's 1 fps sampling is two seconds apart on the shared clock. The [timeline CSV](outputs/youtube/shared_temporal_waste_pit.csv), [incident JSON](outputs/youtube/shared_temporal_waste_pit_incident.json), and [annotated replay](outputs/youtube/5NAkyEmC0IU_shared_temporal_h264.mp4) record the result.

## What the shared context holds

This is a rule-based **offline incident record**, not a learned temporal model or a live alarm. It reads saved model boxes from `outputs/youtube/frame_results.csv`. It holds the known `waste_pit` zone ID; each view's first/last flame, smoke and same-image D-FINE/D-Fire agreement times; models and classes seen; first cross-view flame and smoke matches; the matching margin; alert stage; and notification history. Within each camera, it also tests overlapping D-FINE boxes on consecutive sampled frames. Later smoke observations update the same incident instead of creating another scripted notification.

The alert stages in this trial are: **watch** for a D-FINE flame box in the reviewed zone; **provisional** when D-FINE and D-Fire overlap on the same flame in one view; **two-view candidate** when D-FINE flags flame in both views within the shared-clock margin. Either of the last two conditions can request operator review. Repeated boxes alone remain watch evidence because a static false box can repeat. Smoke cross-view evidence is recorded but does not trigger a new alert after the flame incident. This prototype has no persistent track IDs, motion test, calibrated probabilities, incident expiry, or camera health handling.

## What happened in this clip

All times below are **original clip playback seconds**, with the shared clock expressed as angle 1 seconds. The reviewed regions were selected after seeing this positive event.

| Shared clock | Angle 1 / angle 2 source frames | Evidence and effect |
| ---: | --- | --- |
| 1–3 s | angle 2 56–57 s | Whole-frame D-FINE repeatedly boxes static clutter as flame. There is no matching angle-1 flame box within ±1 s, so the two-view rule does not corroborate it. The post-hoc fire-area filter also excludes those boxes. |
| **5 s** | **5 / 58 s** | D-FINE boxes the visible small flame in both views. This is the first margin-aware two-view flame match. D-Fire also overlaps D-FINE on the same angle-2 flame (box IoU about 0.95). The staged replay issues one provisional review and reaches the two-view candidate stage. |
| 6 s | 6 / about 58.5 s | Angle 1 first has a same-location D-FINE flame repeat; D-FINE and D-Fire first agree there. |
| 7 s | 7 / 59 s | Angle 2 first has a same-location D-FINE flame repeat. |
| **13 s** | **13 / 62 s** | D-FINE first flags pit-zone smoke in both views within the margin. That updates the existing incident; it does not make a second operator notification. |

Thus the two-view rule corroborates the **first** D-FINE flame hits in this case, one shared playback second before angle 1's local repeat and two before angle 2's. It gives **no earlier first operator signal than same-image D-FINE/D-Fire agreement** here: both occur at shared-clock 5 s. The useful addition is evidence from a second viewpoint. The smoke match is also earlier than the first local smoke repeat (angle 1 at 14 s; angle 2 at source 63 s, shared-clock 15 s). Rechecking with matching margins of 0, 1 and 2 shared-clock seconds leaves the first flame match at 5 s and smoke match at 13 s in these saved samples.

The early angle-2 false flame boxes would produce a naive whole-frame two-frame alert at shared-clock 3 s. They do **not** pass cross-view corroboration in this clip, even without the reviewed-area filter. This is a useful failure case, but one short pre-fire segment cannot estimate false-alert reduction. The one notification in the incident JSON is also a direct consequence of the one-incident state machine, not a measured reduction in duplicate notifications.

## Outside-clip check and limits

The [FURG same-camera model-agreement check](outputs/furg/model_agreement_check.csv) provides a caution about making any second-evidence rule mandatory. Among 79 flame-labeled sampled frames in three clips, D-FINE boxes flame in 71 and overlapping D-FINE/D-Fire boxes occur in 61. Among 102 frames from two flame-free clips, D-FINE has three flame hits and the overlap rule has none, but ordinary D-FINE two-frame persistence also makes zero alerts there. On `house1`, D-FINE first hits at 13.013 s, first repeats at 14.014 s, and first overlaps D-Fire only at 19.019 s. The second camera could provide a faster alternative when another model misses, but this edited waste-pit clip cannot establish how often that happens.

This trial uses one positive incident, saved 1 fps predictions, manually chosen regions, and a user-estimated time map. The two views came from one edited video, so camera-clock error, dropped footage and playback changes cannot be independently checked. We can describe **when the candidate rules fire in this case**; we cannot yet measure detection sensitivity, false-alert episodes per camera-hour, or operational speed gains. The next real test needs simultaneous camera exports, fixed camera-to-zone mapping, several fires, and long clear periods with dust, steam, hot work and moving equipment. It should compare single-camera persistence, same-camera model agreement and shared-zone corroboration with a fallback for an occluded or offline camera.
