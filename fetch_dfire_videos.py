"""Fetch a fixed six-clip sample from D-Fire's published video test split."""

from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright
import requests


SHARE_URL = (
    "https://1drv.ms/f/c/c0bd25b6b048b01d/"
    "EhT2Jy6L-YlGvZv-gXH2SnYBENQsnUW96LpZtv_6PngjYQ"
)
SAMPLE = ("FP1.mp4", "FP14.mp4", "FP31.mp4", "VP1.mp4", "VP3.mp4", "VP5.mp4")
SOURCE_FOLDER = (
    "/personal/c0bd25b6b048b01d/Documents/Professional career/Academia/"
    "Research/Fire detection/Videos"
)
DESTINATION = Path(__file__).parent / "dataset" / "videos"


def main() -> None:
    test_names = set(
        (Path(__file__).parent / "dataset" / "splits" / "video_test.txt")
        .read_text(encoding="utf-8-sig").splitlines()
    )
    if not set(SAMPLE) <= test_names:
        raise RuntimeError("Sample includes a clip outside the published video test split")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(SHARE_URL, wait_until="domcontentloaded", timeout=60000)
        page.get_by_text("FP1.mp4", exact=True).first.wait_for(timeout=30000)
        session = requests.Session()
        for cookie in context.cookies():
            session.cookies.set(
                cookie["name"], cookie["value"],
                domain=cookie["domain"], path=cookie["path"],
            )
        browser.close()
    for name in SAMPLE:
        target = DESTINATION / name
        if target.exists():
            print(f"Already present: {name}", flush=True)
            continue
        source = quote(f"{SOURCE_FOLDER}/{name}", safe="/")
        url = (
            "https://onedrive.live.com/personal/c0bd25b6b048b01d/"
            f"_layouts/15/download.aspx?SourceUrl={source}"
        )
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if response.headers.get("Content-Type", "").split(";")[0] != "video/mp4":
                raise RuntimeError(f"Unexpected content type for {name}")
            with target.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        print(f"{name}: {target.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
