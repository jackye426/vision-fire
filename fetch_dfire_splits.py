"""Download the published D-Fire split lists from their public OneDrive folder."""

from pathlib import Path

from playwright.sync_api import sync_playwright


SPLITS_URL = (
    "https://1drv.ms/f/c/c0bd25b6b048b01d/"
    "Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN"
)
SPLIT_FILES = ("dfire_test.txt", "dfire_train.txt", "video_test.txt", "video_valid.txt")
DESTINATION = Path(__file__).parent / "dataset" / "splits"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        for name in SPLIT_FILES:
            page.goto(SPLITS_URL, wait_until="domcontentloaded", timeout=60000)
            page.get_by_text(name, exact=True).wait_for(timeout=30000)
            page.get_by_text(name, exact=True).first.click()
            with page.expect_download(timeout=60000) as download_info:
                page.get_by_text("Download", exact=True).first.click()
            download = download_info.value
            target = DESTINATION / name
            download.save_as(target)
            print(f"{target}: {target.stat().st_size} bytes")
        browser.close()


if __name__ == "__main__":
    main()
