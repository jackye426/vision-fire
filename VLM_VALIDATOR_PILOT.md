# OpenRouter VLM validator pilot — 27 September 2026

We tested whether a vision-language model can check **one detector box** before an alert. This was a small case review of public video frames, not a model benchmark or an EMR-camera test. We used [`google/gemini-3-flash-preview`](https://openrouter.ai/google/gemini-3-flash-preview) through [OpenRouter's image-input chat API](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding).

## Method

- Nine proposals were selected from the saved 1 fps detector output: three visible flame boxes, two visible smoke boxes, two repeats of one static false flame target, one false excavator target, and one early smoke candidate whose human label remains uncertain.
- For each proposal, the VLM saw the original frame and a native-pixel crop with the **proposed box outlined in cyan**. The prompt asked whether the requested phenomenon was visible **inside that box**, not anywhere in the frame. We hid detector name, confidence, answer and later frames.
- On three proposals, we repeated the query with one earlier sampled crop as additional context. These are separate calls, not a video-model inference.
- `openrouter_vlm_pilot.py` verifies every target box against `outputs/youtube/frame_results.csv`, prepares the exact local JPEG inputs, and records model ID, verdict, wording, source time, input hashes, API wall time and OpenRouter usage/cost. The JPEGs and API key stay local and ignored by Git.

## What happened

| Detector proposal | Human review | Single-frame VLM | API time |
| --- | --- | --- | ---: |
| Waste pit angle 2, 56 s, red static lights boxed as flame | No flame in box | No | 2.30 s |
| Same static object, 57 s | No flame in box | No | 2.38 s |
| Waste pit angle 1, 5 s, tiny flame | Flame | Yes | 2.43 s |
| Waste pit angle 2, 58 s, tiny flame | Flame | Yes | 2.30 s |
| Waste pit angle 1, 12 s, subtle possible smoke | **Uncertain** | Yes | 2.54 s |
| Waste pit angle 1, 13 s, smoke | Smoke | Yes | 2.32 s |
| Battery clip, 7 s, excavator boxed as flame | No flame in box | No | 2.44 s |
| Battery clip, 10 s, flame | Flame | Yes | 1.94 s |
| Recycling clip, 10 s, smoke | Smoke | Yes | 2.39 s |

The VLM agreed with all **eight determinate box reviews** in this deliberately selected set. The 56 and 57 s negatives are one repeated error, and the two waste-pit flames are views of one event; this is **not** 8 independent incidents or an accuracy estimate. The model called the early 12 s plume smoke, but that remains an unscored, ambiguous human review. Its description of the red static lights as vehicle tail lights may be more specific than the image warrants, despite the correct no-flame decision.

The single-frame API round trip had median **2.38 s** (range 1.94–2.54 s). This clock starts after local image preparation and excludes detector time, queueing, and operational network conditions. Adding the previous crop did not change any of the three selected verdicts: false object **no**, tiny flame **yes**, early smoke **yes**. Those three calls took 2.11, 2.89, and 2.48 s respectively; this sample does not establish a temporal benefit. OpenRouter reported a combined cost of **$0.0179 for all 12 calls**. Rates and routing can change.

## What this means for alert logic

For this one waste-pit clip, a targeted VLM check could have rejected the early **D-FINE + FireViewer YOLO11-M shared false box**, which a simple two-model vote would have accepted. It did not create an earlier flame detection: the small real flame was already proposed by D-FINE and corroborated at shared playback time 5 s by D-Fire in angle 2 and D-FINE in both views. A blocking VLM call would add an API round trip to that decision.

The 12 s smoke candidate is the interesting open question. The VLM said yes one edited playback second before the saved cross-model smoke agreement at 13 s. Because the label is visually subtle, the two sections of the source clip have altered speeds, and the API call took about 2.5 wall-clock seconds, we cannot claim faster real-world smoke verification. A human frame-by-frame label and original camera timestamps are needed.

Next, keep the VLM in **shadow mode** on a fixed sample of independent fire incidents and long clear-camera periods. Review target boxes blind to the VLM answer; count true rejected false proposals, true suppressed fires, uncertain outputs, cost, and alert delay. Include dust, steam, machine lights, and equipment, and compare against the existing same-location and cross-view rules. Only then decide whether a VLM should block or merely annotate an operator alert.

## Reproduce

Place an OpenRouter key in ignored `.env.local` as `OPENROUTER_API_KEY=...`, or set that environment variable. With the existing public clips and saved detector CSV available locally:

```powershell
python openrouter_vlm_pilot.py                    # prepare/inspect JPEGs; no API call
python openrouter_vlm_pilot.py --run              # nine single-frame calls
python openrouter_vlm_pilot.py --run --temporal   # three prior-frame comparisons
```

Recorded responses: [single-frame JSONL](outputs/vlm_validator/single_results.jsonl) and [prior-frame JSONL](outputs/vlm_validator/temporal_results.jsonl). The script skips already completed case/model pairs on rerun.
