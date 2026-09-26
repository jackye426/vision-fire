"""Local comparison UI for pretrained smoke and flame detectors."""

from io import BytesIO
from pathlib import Path
import tempfile
from time import perf_counter
import zipfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from detector import SPECS, annotate, detect, session

ROOT = Path(__file__).resolve().parent


def video_frames(data: bytes, suffix: str, interval: float, max_frames: int):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video. Try MP4/H.264 or MOV.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            raise ValueError("This video has no readable frame rate or frame count.")
        stride = max(1, round(interval * fps))
        frames = []
        for index in range(0, total, stride):
            if len(frames) >= max_frames:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((1280, 1280))
                frames.append((round(index / fps, 2), image))
        capture.release()
        if not frames:
            raise ValueError("No video frames could be decoded.")
        return frames, round(total / fps, 2)
    finally:
        path.unlink(missing_ok=True)


def first_persistent(times, flags, required):
    streak = 0
    for index, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak >= required:
            return times[index]  # alert is knowable only when the streak is confirmed
    return None


def count_alert_episodes(flags, required):
    streak = 0
    episodes = 0
    for flag in flags:
        streak = streak + 1 if flag else 0
        if streak == required:
            episodes += 1
    return episodes


def summary(run, threshold, required):
    rows = []
    for name in run["models"]:
        frames = run["results"][name]
        times = [frame["time"] for frame in frames]
        smoke = [any(d.label == "smoke" and d.score >= threshold for d in frame["detections"]) for frame in frames]
        flame = [any(d.label == "flame" and d.score >= threshold for d in frame["detections"]) for frame in frames]
        any_signal = [s or f for s, f in zip(smoke, flame)]
        raw_hit = first_persistent(times, any_signal, 1)
        alert = first_persistent(times, any_signal, required)
        rows.append({
            "Model": SPECS[name]["title"],
            "First smoke (s)": first_persistent(times, smoke, required),
            "First flame (s)": first_persistent(times, flame, required),
            "First raw hit (s)": raw_hit,
            "First persistent alert (s)": alert,
            "Persistence delay (s)": round(alert - raw_hit, 2) if alert is not None and raw_hit is not None else None,
            "Flagged frames": sum(any_signal),
            "Sampled frames": len(frames),
            "Mean CPU inference (ms)": round(1000 * sum(run["latencies"][name]) / len(frames)),
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Smoke & fire model lab", page_icon="🔥", layout="wide")
st.title("Smoke & fire model lab")
st.caption("Explore pretrained RGB detectors on still images and sampled video frames. Runs locally on CPU.")

with st.sidebar:
    st.header("Trial settings")
    available_models = [name for name, spec in SPECS.items() if spec["path"].exists()]
    default_models = [name for name in ("dfire_yolov8n", "soul", "pyronear") if name in available_models]
    models = st.multiselect("Models", available_models, default=default_models,
                            format_func=lambda name: SPECS[name]["title"])
    threshold = st.slider("Detection score threshold", 0.05, 0.95, 0.25, 0.05,
                          help="Filter model scores after inference. Scores are not calibrated fire probabilities.")
    interval = st.number_input("Video sample interval (seconds)", 0.25, 30.0, 2.0, 0.25)
    max_frames = st.number_input("Maximum video frames", 1, 100, 20, 1)
    required = st.number_input("Consecutive sampled frames for alert", 1, 10, 2, 1)
    st.caption("An alert time is recorded on the frame that confirms the streak. Set this to 1 to inspect raw per-frame detections.")
    st.divider()
    st.markdown("[EMR visit notes](https://app.notion.com/p/3e5196f5637381aea9c8e09aeb82013a) · [D-Fire](https://huggingface.co/rabahdev/fire-smoke-yolov8n) · [SoulPerforms](https://huggingface.co/spaces/SoulPerforms/Fire_Detection_YOLOv8_Model_Inference_with_Gradio) · [Pyronear](https://huggingface.co/pyronear/yolo11s_sensitive-detector) · [D-FINE](https://huggingface.co/fireviewer/fire-smoke-dfine-m-strict-v1) · [YOLO11-M](https://huggingface.co/fireviewer/fire-smoke-yolo11m-strict-v1)")

with st.expander("What kinds of setup are these?", expanded=False):
    st.markdown("""
    - **Single-frame fire + smoke:** D-Fire YOLOv8n, D-FINE, and FireViewer YOLO11-M locate visible fire or smoke in each RGB frame. They need temporal filtering for monitoring.
    - **Single-frame fire only:** the SoulPerforms Space's actual ONNX checkpoint has one class, `fire`; it cannot detect smoke before flame appears.
    - **Single-frame smoke specialist:** Pyronear targets distant wildfire plumes. It can miss flames and may transfer poorly to scrap yards.
    - **Temporal systems:** the [pedbrgs research system](https://github.com/pedbrgs/Fire-Detection) combines YOLOv5 candidates with area-variation or persistence analysis. [PyroNear's temporal smoke model](https://huggingface.co/pyronear/temporal-model) links YOLO boxes into tubes and classifies their frame patches with a vision encoder and transformer. The streak setting here is a simple persistence experiment, **not** a reproduction of either system.
    - **Two specialists together:** the measured smoke-specialist + fire-specialist row below combines PyroNear Sensitive and SoulPerforms detections with an OR rule. It runs two models and is not a trained fusion system.
    - **Hosted model:** [Roboflow's community model](https://universe.roboflow.com/s-workspace-173hq/fire-smoke-detection-avnev-goqj2) is a separate YOLO26 fire/smoke model. Its hosted API requires a key; uploading footage there sends it to a third party.
    """)

image_summary_path = ROOT / "outputs" / "dfire_benchmark" / "image_summary.csv"
video_summary_path = ROOT / "outputs" / "dfire_benchmark" / "video_summary.csv"
localization_path = ROOT / "outputs" / "dfire_benchmark" / "localization_results.csv"
pair_image_path = ROOT / "outputs" / "dfire_benchmark" / "specialist_pair_image.csv"
pair_video_path = ROOT / "outputs" / "dfire_benchmark" / "specialist_pair_video.csv"
temporal_path = ROOT / "outputs" / "dfire_benchmark" / "pyronear_temporal_windows.csv"
rule_sweep_path = ROOT / "outputs" / "dfire_benchmark" / "alert_rule_sweep.csv"
if image_summary_path.exists() and video_summary_path.exists():
    with st.expander("Measured D-Fire comparison", expanded=True):
        st.caption("160 original test images: 40 each with no label, smoke only, flame only, and both. A hit means any box with the right class, regardless of its location. This balanced sample is exploratory. D-Fire YOLOv8n was trained on this dataset family, so its result is not a transfer test.")
        benchmark_threshold = st.selectbox("Benchmark score threshold", [0.25, 0.50])
        measured = pd.read_csv(image_summary_path)
        selected = measured[measured["threshold"] == benchmark_threshold]
        comparison = []
        for model in SPECS:
            model_rows = selected[selected["model"] == model]
            if model_rows.empty:
                continue
            smoke_row = model_rows[model_rows["class"] == "smoke"]
            flame_row = model_rows[model_rows["class"] == "flame"]
            def hit_text(rows):
                if rows.empty:
                    return "Not supported"
                hit = int(rows.iloc[0]["tp"])
                return f"{hit}/80 ({hit / 80:.0%})"
            comparison.append({
                "Model": SPECS[model]["title"],
                "Smoke images hit": hit_text(smoke_row),
                "Flame images hit": hit_text(flame_row),
                "Clear images flagged": f"{int(model_rows.iloc[0]['clear_image_hits'])}/40",
                "Median CPU ms/image": float(model_rows.iloc[0]["median_cpu_ms"]),
            })
        if pair_image_path.exists():
            pair = pd.read_csv(pair_image_path)
            pair_rows = pair[pair["threshold"] == benchmark_threshold]
            if not pair_rows.empty:
                row = pair_rows.iloc[0]
                comparison.append({
                    "Model": "PyroNear smoke OR SoulPerforms fire",
                    "Smoke images hit": f"{int(row['smoke_hits_of_80'])}/80 ({row['smoke_hits_of_80'] / 80:.0%})",
                    "Flame images hit": f"{int(row['flame_hits_of_80'])}/80 ({row['flame_hits_of_80'] / 80:.0%})",
                    "Clear images flagged": f"{int(row['clear_flags_of_40'])}/40",
                    "Median CPU ms/image": float(row["median_cpu_ms_per_image"]),
                })
        st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
        if localization_path.exists():
            localized = pd.read_csv(localization_path)
            localized_rows = []
            for model in SPECS:
                group = localized[localized["model"] == model]
                counts = group.groupby("class")["localized_iou50"].sum().to_dict()
                localized_rows.append({
                    "Model": SPECS[model]["title"],
                    "Smoke boxes aligned": f"{int(counts['smoke'])}/80" if "smoke" in counts else "Not supported",
                    "Flame boxes aligned": f"{int(counts['flame'])}/80" if "flame" in counts else "Not supported",
                })
            if pair_image_path.exists():
                pair = pd.read_csv(pair_image_path)
                pair_rows = pair[pair["threshold"] == 0.25]
                if not pair_rows.empty:
                    row = pair_rows.iloc[0]
                    localized_rows.append({
                        "Model": "PyroNear smoke OR SoulPerforms fire",
                        "Smoke boxes aligned": f"{int(row['smoke_aligned_iou50_of_80_at_025'])}/80",
                        "Flame boxes aligned": f"{int(row['flame_aligned_iou50_of_80_at_025'])}/80",
                    })
            st.caption("At score 0.25, images with a same-class prediction overlapping a labeled box by at least 50%:")
            st.dataframe(pd.DataFrame(localized_rows), hide_index=True, width="stretch")
        st.caption("The exact sampling and scoring protocol is in BENCHMARK_REPORT.md. Model scores are not calibrated to one another.")
        videos = pd.read_csv(video_summary_path)
        clip = st.selectbox("Inspect a measured test clip", sorted(videos["video"].unique()))
        clip_rows = videos[(videos["video"] == clip) &
                           (videos["threshold"] == benchmark_threshold)].copy()
        clip_rows["Model"] = clip_rows["model"].map(lambda name: SPECS[name]["title"])
        if pair_video_path.exists():
            pair = pd.read_csv(pair_video_path)
            pair_rows = pair[(pair["video"] == clip) & (pair["threshold"] == benchmark_threshold)]
            if not pair_rows.empty:
                row = pair_rows.iloc[0]
                clip_rows = pd.concat([clip_rows, pd.DataFrame([{
                    "Model": "PyroNear smoke OR SoulPerforms fire",
                    "sampled_frames": row["sampled_frames"],
                    "flagged_frames": row["flagged_frames"],
                    "first_raw_s": None,
                    "first_2_hit_s": row["first_2_hit_s"],
                    "alert_episodes_2_hit": None,
                }])], ignore_index=True)
        st.dataframe(clip_rows[["Model", "sampled_frames", "flagged_frames", "first_raw_s",
                                "first_2_hit_s", "alert_episodes_2_hit"]],
                     hide_index=True, width="stretch")
        st.caption("Clips were sampled at about one frame per second. The published clip list lacks frame-level labels and smoke-onset times, so detection timing is relative to the clip start, not measured lead time.")
        if rule_sweep_path.exists():
            policies = pd.read_csv(rule_sweep_path)
            policy_rows = policies[(policies["video"] == clip) &
                                   (policies["threshold"] == benchmark_threshold)].copy()
            if not policy_rows.empty:
                policy_rows["Model"] = policy_rows["model"].map(lambda name: SPECS[name]["title"])
                policy_table = policy_rows.pivot(index="Model", columns="rule", values="first_alert_s")
                policy_table = policy_table.reindex(columns=["1_of_1_same_class", "2_of_2_same_class",
                                                             "3_of_3_same_class", "2_of_3_same_class"])
                policy_table.columns = ["1 hit (s)", "2 consecutive (s)",
                                        "3 consecutive (s)", "2 of last 3 (s)"]
                st.markdown("#### Alert-rule replay")
                st.dataframe(policy_table.reset_index(), hide_index=True, width="stretch")
                st.caption("First candidate time for each rule using saved 1 fps frame scores. Each rule requires the same class across hits, but does not yet track box location. Blank means no alert. These are not verified event detection times.")
        if temporal_path.exists():
            temporal = pd.read_csv(temporal_path)
            temporal_rows = temporal[temporal["video"] == clip]
            if not temporal_rows.empty:
                st.markdown("#### PyroNear temporal smoke model")
                st.dataframe(temporal_rows[["window_start_s", "window_end_s", "frames",
                                            "detector_hits", "kept_tubes", "max_tube_probability",
                                            "smoke_decision", "tube_pipeline_ms"]],
                             hide_index=True, width="stretch")
                st.caption("This is PyroNear's packaged companion YOLO plus tube classifier, evaluated on up to 20 frames per window at about 1 fps. Its window verdict is a different metric from the two-hit frame alert above; detector time is measured separately and is reported in the CSV.")

furg_summary_path = ROOT / "outputs" / "furg" / "summary.csv"
cmu_aggregate_path = ROOT / "outputs" / "cmu" / "aggregate.csv"
if furg_summary_path.exists() and cmu_aggregate_path.exists():
    with st.expander("Measured results outside D-Fire", expanded=True):
        st.caption("Small transfer checks only: five FURG flame videos and six CMU industrial-smoke clips. Read CROSS_DATASET_REPORT.md for sampling, labels and limitations. The score cutoff is applied to each model's uncalibrated output.")
        external_threshold = st.selectbox("External-data score threshold", [0.25, 0.50])
        furg = pd.read_csv(furg_summary_path)
        furg_rows = furg[furg["threshold"] == external_threshold]
        furg_table = []
        for model in SPECS:
            matches = furg_rows[furg_rows["model"] == model]
            if matches.empty:
                continue
            row = matches.iloc[0]
            supports_flame = "flame" in SPECS[model]["labels"]
            furg_table.append({
                "Model": SPECS[model]["title"],
                "Flame frames hit": (f"{int(row['flame_hit_frames'])}/79"
                                     if supports_flame else "Not supported"),
                "Flame boxes aligned": (f"{int(row['flame_aligned_iou50_frames'])}/79"
                                        if supports_flame else "Not supported"),
                "Flame flags on clear frames": (f"{int(row['clear_clip_flame_flags'])}/102"
                                                if supports_flame else "Not supported"),
                "Clear clips with two-hit flame": (f"{int(row['clear_clips_with_2_hit_flame'])}/2"
                                                  if supports_flame else "Not supported"),
                "Median CPU ms/frame": float(row["median_cpu_ms_per_frame"]),
            })
        st.markdown("#### FURG: flame localization on five videos")
        st.dataframe(pd.DataFrame(furg_table), hide_index=True, width="stretch")
        st.caption("79 sampled frames have FURG flame boxes; 102 frames come from two flame-free clips. Images within one video are correlated. FURG has no smoke annotations.")
        cmu = pd.read_csv(cmu_aggregate_path)
        cmu_rows = cmu[cmu["threshold"] == external_threshold]
        cmu_table = []
        for model in SPECS:
            if "smoke" not in SPECS[model]["labels"]:
                continue
            matches = cmu_rows[cmu_rows["model"] == model]
            if matches.empty:
                continue
            row = matches.iloc[0]
            cmu_table.append({
                "Model": SPECS[model]["title"],
                "Smoke clips flagged": f"{int(row['positive_clips_with_smoke_2hit_of_3'])}/3",
                "Clear clips flagged": f"{int(row['clear_clips_with_smoke_2hit_of_3'])}/3",
                "Median CPU ms/frame": float(row["median_cpu_ms_per_frame"]),
            })
        st.markdown("#### CMU: industrial-smoke clip decisions")
        st.dataframe(pd.DataFrame(cmu_table), hide_index=True, width="stretch")
        st.caption("Each clip has 36 time-lapse frames. Frame detectors require two consecutive smoke-box frames. PyroNear's released temporal package uses its own detector and packaged threshold: 1/3 smoke clips and 0/3 clear clips at either displayed frame-detector cutoff. Clear-labelled CMU clips can show pale steam-like plumes; there are no frame-level smoke boxes or onset times.")

fire_demo_path = ROOT / "outputs" / "furg" / "hand_held_camera_wildfire_dfire_yolov8n_h264.mp4"
if fire_demo_path.exists():
    with st.expander("Watch D-Fire YOLOv8n on an independent vehicle-fire clip", expanded=True):
        st.caption("Public [FURG fire video](https://github.com/steffensbola/furg-fire-dataset), sampled at ~10 fps. Orange boxes are model flame detections; green boxes are model smoke detections at score 0.25. This is one visual example, not a cross-dataset performance estimate.")
        st.video(str(fire_demo_path))

source = st.radio("Input", ["Official D-Fire test image", "Official D-Fire test video",
                            "Independent FURG fire video",
                            "Upload image or video", "Included D-Fire examples"], horizontal=True)
uploaded = st.file_uploader("Choose JPG, PNG, MP4 or MOV", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"],
                            disabled=source != "Upload image or video")
sample_path = ROOT / "samples" / "dfire_preview.jpg"
archive_path = ROOT / "dataset" / "D-Fire.zip"
manifest_path = ROOT / "outputs" / "dfire_benchmark" / "image_manifest.csv"
video_folder = ROOT / "dataset" / "videos"
selected_image = None
selected_video = None
if source == "Official D-Fire test image":
    if archive_path.exists() and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        category = st.selectbox("Ground-truth category", ["none", "smoke_only", "flame_only", "both"],
                                format_func=lambda value: value.replace("_", " ").title())
        choices = manifest.loc[manifest["category"] == category, "name"].tolist()
        selected_image = st.selectbox("Test image", choices)
        st.caption("Original image and YOLO label from the published 4,306-image D-Fire test split. Choose a category and inspect model boxes.")
    else:
        st.info("Run fetch_dfire_archive.py and benchmark_dfire_images.py to enable the official test images.")
elif source == "Official D-Fire test video":
    choices = [name for name in ("FP1.mp4", "FP14.mp4", "FP31.mp4", "VP1.mp4", "VP3.mp4", "VP5.mp4")
               if (video_folder / name).exists()]
    if choices:
        selected_video = st.selectbox("Test video", choices)
        st.caption("These clips are in the published video test list. It has no frame-level labels or event-onset timestamps, so inspect the frames as well as the scores.")
    else:
        st.info("Run fetch_dfire_videos.py to enable the official test clips.")
elif source == "Independent FURG fire video":
    selected_video = "hand_held_camera_wildfire.mp4"
    st.caption("Independent [FURG fire dataset](https://github.com/steffensbola/furg-fire-dataset) clip with visible vehicle flames. Its published XML marks flame rectangles, but this app's interactive output shows model detections only.")
available = ((source == "Included D-Fire examples" and sample_path.exists())
             or (source == "Official D-Fire test image" and selected_image is not None)
             or (source == "Official D-Fire test video" and selected_video is not None)
             or (source == "Independent FURG fire video" and
                 (ROOT / "dataset" / "furg" / "hand_held_camera_wildfire.mp4").exists())
             or (source == "Upload image or video" and uploaded is not None))

if source == "Included D-Fire examples":
    st.info("This is an annotated collage from the D-Fire dataset. It checks that the models run; it is not an independent benchmark or an EMR camera view.")

if st.button("Run comparison", type="primary", disabled=not available or not models):
    try:
        if source == "Included D-Fire examples":
            image = Image.open(sample_path).convert("RGB")
            image.thumbnail((1280, 1280))
            frames, duration, name = [(0.0, image)], 0.0, sample_path.name
        elif source == "Official D-Fire test image":
            name = selected_image
            with zipfile.ZipFile(archive_path) as archive:
                data = archive.read(f"test/images/{name}")
            image = Image.open(BytesIO(data)).convert("RGB")
            image.thumbnail((1280, 1280))
            frames, duration = [(0.0, image)], 0.0
        elif source == "Official D-Fire test video":
            name = selected_video
            frames, duration = video_frames((video_folder / name).read_bytes(), ".mp4",
                                             float(interval), int(max_frames))
        elif source == "Independent FURG fire video":
            name = selected_video
            frames, duration = video_frames((ROOT / "dataset" / "furg" / name).read_bytes(),
                                             ".mp4", float(interval), int(max_frames))
        else:
            name = uploaded.name
            data = uploaded.getvalue()
            suffix = Path(name).suffix.lower()
            if suffix in (".mp4", ".mov", ".avi"):
                frames, duration = video_frames(data, suffix, float(interval), int(max_frames))
            else:
                image = Image.open(BytesIO(data)).convert("RGB")
                image.thumbnail((1280, 1280))
                frames, duration = [(0.0, image)], 0.0
        results = {model: [] for model in models}
        latencies = {model: [] for model in models}
        progress = st.progress(0, text="Loading models and examining frames…")
        for model in models:
            session(model)
        total = len(frames) * len(models)
        done = 0
        for time, image in frames:
            for model in models:
                started = perf_counter()
                found = detect(model, image)
                latencies[model].append(perf_counter() - started)
                results[model].append({"time": time, "detections": found})
                done += 1
                progress.progress(done / total, text=f"Processed {done}/{total} model frames")
        progress.empty()
        st.session_state["run"] = {
            "name": name, "models": models, "frames": frames,
            "results": results, "duration": duration,
            "latencies": latencies,
            "interval": float(interval),
        }
    except Exception as exc:
        st.error(f"Comparison failed: {exc}")

run = st.session_state.get("run")
if run:
    st.subheader(f"Results · {run['name']}")
    if run["duration"]:
        st.caption(f"Video duration {run['duration']:.1f}s · sampled {len(run['frames'])} frames through {run['frames'][-1][0]:.1f}s every ~{run['interval']:.2f}s. Intervals between samples were not inspected.")
        if run["frames"][-1][0] + run["interval"] < run["duration"]:
            st.warning("The frame cap stopped analysis before the video ended. Increase the maximum frames or sample interval to cover more of the clip.")
    else:
        st.caption("One still image; temporal persistence is not applicable.")
    summary_table = summary(run, threshold, int(required) if run["duration"] else 1)
    st.dataframe(summary_table, hide_index=True, width="stretch")

    if run["duration"]:
        if st.checkbox("This entire clip is known to be clear", help="Count model alerts on footage with no fire or smoke. One continuous run counts as one episode."):
            negative_rows = []
            for model in run["models"]:
                flags = [any(d.score >= threshold for d in item["detections"])
                         for item in run["results"][model]]
                negative_rows.append({"Model": SPECS[model]["title"],
                                      "False alert episodes": count_alert_episodes(flags, int(required)),
                                      "Flagged sampled frames": sum(flags)})
            st.dataframe(pd.DataFrame(negative_rows), hide_index=True, width="stretch")
        human_time = st.number_input("Optional human escalation time (seconds into clip)", min_value=0.0,
                                     max_value=float(run["duration"]), value=0.0, step=1.0,
                                     help="Enter a known escalation time to compare lead time. Leave at 0 if unknown.")
        if human_time > 0:
            for _, row in summary_table.iterrows():
                if pd.notna(row["First persistent alert (s)"]):
                    lead = human_time - float(row["First persistent alert (s)"])
                    st.write(f"{row['Model']}: **{lead:+.1f}s** lead relative to entered escalation time")
        rows = []
        for model in run["models"]:
            for item in run["results"][model]:
                eligible = [d for d in item["detections"] if d.score >= threshold]
                rows.append({"time_s": item["time"], "model": SPECS[model]["title"],
                             "smoke_score": max([d.score for d in eligible if d.label == "smoke"], default=0),
                             "flame_score": max([d.score for d in eligible if d.label == "flame"], default=0),
                             "detections": len(eligible)})
        timeline = pd.DataFrame(rows)
        st.markdown("#### Detection timeline")
        st.dataframe(timeline, hide_index=True, width="stretch")
        st.download_button("Download frame scores as CSV", timeline.to_csv(index=False),
                           file_name="fire_smoke_frame_scores.csv", mime="text/csv")
        frame_index = st.slider("Inspect sampled frame", 0, len(run["frames"]) - 1, 0)
    else:
        frame_index = 0

    time, image = run["frames"][frame_index]
    if run["duration"]:
        st.markdown(f"#### Frame at {time:.2f}s")
    columns = st.columns(len(run["models"]))
    for column, model in zip(columns, run["models"]):
        with column:
            detections = run["results"][model][frame_index]["detections"]
            filtered = [d for d in detections if d.score >= threshold]
            st.markdown(f"**{SPECS[model]['title']}**")
            st.image(annotate(image, detections, threshold), width="stretch")
            if filtered:
                st.caption(", ".join(f"{d.label} {d.score:.2f}" for d in filtered[:5]))
            else:
                st.caption("No detection at this threshold")

st.divider()
st.caption("Research prototype for human review. RGB models cannot see hidden or occluded heat, and thermal camera colours are not physical temperatures. Test on independent overnight footage and ordinary false-positive scenes before drawing operational conclusions.")
