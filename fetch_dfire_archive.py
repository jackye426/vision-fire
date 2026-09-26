"""Resumably download the original D-Fire image and annotation ZIP.

OneDrive first requires visiting the public sharing page in a browser to redeem
its anonymous access cookie. The actual transfer uses HTTP Range requests.
"""

from pathlib import Path
import time

from playwright.sync_api import sync_playwright
import requests


SHARE_URL = (
    "https://1drv.ms/u/c/c0bd25b6b048b01d/"
    "EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw"
)
DOWNLOAD_URL = (
    "https://onedrive.live.com/personal/c0bd25b6b048b01d/_layouts/15/download.aspx"
    "?SourceUrl=%2Fpersonal%2Fc0bd25b6b048b01d%2FDocuments%2FProfessional%20career"
    "%2FAcademia%2FResearch%2FFire%20detection%2FD%2DFire%20dataset%2FD%2DFire%2Ezip"
)
DESTINATION = Path(__file__).parent / "dataset" / "D-Fire.zip"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    partial = DESTINATION.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(SHARE_URL, wait_until="domcontentloaded", timeout=60000)
        page.get_by_text("D-Fire.zip", exact=True).first.wait_for(timeout=30000)
        session = requests.Session()
        for cookie in context.cookies():
            session.cookies.set(
                cookie["name"], cookie["value"],
                domain=cookie["domain"], path=cookie["path"],
            )
        browser.close()

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            raise RuntimeError("Server did not honor Range; refusing to append to partial file")
        size = int(response.headers.get("Content-Length", "0")) + offset
        if size < 3_000_000_000:
            raise RuntimeError(f"Unexpected archive size: {size}")
        print(f"Downloading {size:,} bytes, resuming at {offset:,}", flush=True)
        last_report = time.monotonic()
        downloaded = offset
        with partial.open("ab") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                target.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report > 20:
                    print(f"{downloaded / size:.1%} ({downloaded:,} bytes)", flush=True)
                    last_report = time.monotonic()
    if partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download: {partial.stat().st_size:,} of {size:,}")
    partial.rename(DESTINATION)
    print(f"Saved {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
