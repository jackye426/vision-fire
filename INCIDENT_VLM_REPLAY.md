# Incident-level VLM replay — 28 September 2026

## Question

Can a vision-language model (VLM) reason over a shared history from D-Fire YOLOv8n, FireViewer D-FINE and FireViewer YOLO11-M to alert earlier or reject more false alarms than (a) the current rule logic and (b) a simpler VLM that checks a few current images?

This is a **fixed-checkpoint, offline case study**, not a full-clip benchmark. We tested 11 decision points across one edited two-angle waste-pit fire, one independent FURG house-fire clip, and one flame-free FURG robot clip. The same [`google/gemini-3-flash-preview`](https://openrouter.ai/google/gemini-3-flash-preview) checkpoint handled both VLM policies through OpenRouter. Checkpoints and target boxes were fixed in `incident_vlm_replay.py` before the incident calls.

## What the shared history contains

At each checkpoint, the incident policy received only detections already available on that clip's playback clock. Its structured packet held the current detector box and class; model overlap at that location; models that disagreed; earlier hits at the same location; recent boxes with camera, model, class, score, coordinates and time; cumulative first/last times and hit counts; available cross-view observations; physical-zone links; and its earlier watch/escalate/update actions. It also saw the same current full frame, marked target crop, and available other-view frame as the simpler VLM. **The simpler VLM saw those images without the history packet.**

The waste-pit view link uses the user's two visual landmarks and an estimated ±1 shared-playback-second margin. Only boxes inside previously reviewed waste-pit regions are marked as the same physical zone; raw same-class boxes in the other camera are listed separately. These regions were chosen after viewing this positive clip, so they are illustrative and cannot be used to claim independent performance. The FURG clips have one camera each. No human label, fire-onset time, future frame or later detector result was sent to either VLM.

The incident VLM could choose `watch`, `request_other_view`, `escalate` or `update_incident`. For the simpler verifier, a first `yes` to flame started one review alert; subsequent `yes` answers updated that same incident. The strict rule baseline is D-FINE/D-Fire same-location flame agreement, with the existing D-FINE cross-view rule additionally available on the waste-pit clip. Applying that rule to the FURG house clip is a comparison policy; the original shared-context prototype was built only for the waste pit. We also scanned all saved frames with a **naive whole-frame any-two-of-three model vote** as a useful diagnostic.

## Results on the selected decision points

| Clip | Approximate visible/labeled flame onset | Strict rule | Naive any two models | Few-image VLM | Incident VLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Waste pit, two edited views | ~2 s shared playback clock | **5 s** | **1 s, false box** | **5 s** | **5 s** |
| FURG `house1` | 5 s FURG flame label | **19 s** | **16 s** | **13 s** | **13 s** |
| FURG `coolerbot`, flame-free | None | None | None | None | None |

The house-fire alert is six **source playback seconds** earlier than the strict two-model rule, and three earlier than the naive any-two vote, for both VLM approaches. It is the first D-FINE flame proposal; the history-aware model did **not** beat the few-image verifier. On the waste pit, both VLMs rejected the false red-light target at angle-2 source seconds 56 and 57, alerted on the real small flame at shared playback second 5, and later updated the same incident with smoke. The strict rule also first alerted at 5 s and avoided the false box. The naive any-two vote would have generated one false incident alert at shared playback second 1 from D-FINE and YOLO11-M boxing the same red static object. On the robot clip, both VLMs correctly rejected the fire-hydrant and yellow-decoration flame boxes at the three selected checkpoints; both rules had no flame alert. The FURG source provides flame labels, not smoke ground truth, so this negative result is **flame only**.

Both VLMs called the subtle waste-pit smoke box at 12 s positive; our human review still marks that one **uncertain**. It cannot count as an early correct smoke finding. `request_other_view` was never selected, so this replay does not test whether requesting a camera helps. The VLM descriptions sometimes guessed the identity of objects (for example, calling red static lights vehicle tail lights). Their correct yes/no box decisions do not validate those extra details.

| Policy | 11 API calls: median measured round trip | 11-call OpenRouter cost | Median input tokens |
| --- | ---: | ---: | ---: |
| Current-image verifier | 2.63 s | $0.0175 | 2,411 |
| Incident history + images | 3.46 s | $0.0245 | 3,969 |

The history-aware version cost about **40% more** for this set and had a longer observed median API round trip. Calls were sequential, not randomized or repeated, and provider latency varied (one box call took 9.23 s; one incident call took 12.46 s). This is a pilot observation, not a latency benchmark. Detector runtime, encoding, queueing and live camera transport are excluded. Source playback seconds and API wall time must remain separate, especially for the speed-altered waste-pit edit.

## Interpretation

The **shared record is useful** for carrying one incident across cameras and model outputs, recording disagreements, linking the two views where the camera-zone map permits, and avoiding repeat notifications. In this trial, giving that record to the VLM did **not** improve the first correct alert or the selected false-alert count beyond a much simpler image verifier. The few-image verifier delivered the only observed speed gain over the two-model rule, on `house1` at 13 rather than 19 source seconds. The extra text history did not create that gain.

A practical next architecture is a deterministic incident record that invokes the small VLM **only for unresolved first proposals or conflicts**: for example, the red-light agreement between D-FINE and YOLO11-M, or D-FINE's lone early `house1` flame. A strong same-location D-FINE/D-Fire or valid cross-view confirmation can proceed by rule; later identical rejected boxes can reuse the stored decision until their appearance changes. Send the full history to a VLM only when the compact check remains uncertain or views contradict each other. This routing strategy is a proposal derived from this replay, **not another measured policy**.

## Limits and next test

Eleven hand-selected VLM decision points across three clips cannot estimate sensitivity, missed incidents, or false-alert episodes per camera-hour. The two rule baselines scan all saved frames, while the VLM policies were called only at the fixed checkpoints; their opportunities therefore differ. The waste-pit time alignment and camera zones are retrospective. We need a fixed, blind set of independent positive incidents and long flame-free periods, with original camera timestamps and verified onset labels. Run all policies on the **same automatically selected proposals**, preserve one notification per physical incident, record wrong-target escalations separately from genuinely false incident alerts, and compare alert availability after model and API latency. Include clips where one camera is occluded or offline, and ask whether `request_other_view` actually retrieves useful evidence.

## Reproduce and inspect

With local source clips, saved detector CSVs, and the ignored `.env.local` OpenRouter key:

```powershell
python incident_vlm_replay.py                # prepare causal packets/images only
python incident_vlm_replay.py --run --policy box
python incident_vlm_replay.py --run --policy incident
python summarize_incident_vlm.py
```

The script skips completed API calls on rerun. The [decision records](outputs/incident_vlm/decisions.jsonl), [event-level comparison](outputs/incident_vlm/comparison.csv), and [exact causal history packets](outputs/incident_vlm/packets/) are committed. Source video and prepared JPEG inputs remain local and ignored by Git; the API key is never written to the results.
