"""Convert the OpenCV demo MP4 to browser-playable H.264.

Requires imageio-ffmpeg in the Python environment used to run this script.
"""

from pathlib import Path
import subprocess

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent / "outputs" / "furg"
SOURCE = ROOT / "hand_held_camera_wildfire_dfire_yolov8n.mp4"
TARGET = ROOT / "hand_held_camera_wildfire_dfire_yolov8n_h264.mp4"


if __name__ == "__main__":
    if not SOURCE.exists():
        raise FileNotFoundError("Run python render_furg_fire.py first")
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(SOURCE), "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(TARGET),
    ], check=True)
    print(f"Saved {TARGET} ({TARGET.stat().st_size:,} bytes)")
