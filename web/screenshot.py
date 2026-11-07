"""Dev tool: screenshot every dashboard view with headless Chromium (Playwright).

    python -m web.screenshot            # writes web/screenshots/*.png (git-ignored)

Starts the local server on a free port in this process (127.0.0.1 only), opens the dashboard,
steps through the views and reports any browser console errors. The browser lives in the
project's .playwright/ folder.
"""
import os
import socket
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))

from playwright.sync_api import sync_playwright  # noqa: E402

from web.server import make_server  # noqa: E402

OUT = ROOT / "web" / "screenshots"
VIEWPORT = {"width": 1440, "height": 900}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(page, extra_ms=1800):
    page.wait_for_function("window.__app && document.body.dataset.ready === '1'", timeout=60000)
    page.wait_for_timeout(extra_ms)  # basemap tiles and the image fade


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    httpd = make_server(port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/"
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        def shot(name, hash_=None, setup=None, wait=1800):
            if hash_ is not None:
                page.goto("about:blank")
                page.goto(f"{base}#{hash_}")
                wait_ready(page, wait)
            if setup:
                setup()
                page.wait_for_timeout(wait)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(f"  saved {name}.png")

        shot("01_map_nakuru_rain", "site=nakuru&date=2025-04-15&h=1&layer=rain1&view=forecast&mode=smooth")
        shot("02_hover", setup=lambda: page.mouse.move(760, 430), wait=600)
        shot("03_drawer", setup=lambda: page.evaluate("window.__app.selectPixel(Math.floor(window.__app.state.pixels.nakuru.ids.length * 0.55))"), wait=2500)
        shot("04_busia_rain", "site=busia&date=2025-08-17&h=1&layer=rain1&view=forecast&mode=smooth")
        shot("04b_busia_humidity_error", "site=busia&date=2025-08-17&h=2&layer=rh&view=error&mode=smooth")
        shot("05_nakuru_blight_cells", "site=nakuru&date=2025-04-15&h=2&layer=blight_hutton&view=forecast&mode=cells")
        shot("06_observed_rain", "site=nakuru&date=2025-04-15&h=1&layer=rain1&view=observed&mode=smooth")
        page.goto("about:blank")
        page.goto(f"{base}#site=nakuru&date=2025-04-15&h=1&layer=rain1&view=forecast")
        wait_ready(page, 800)
        shot("07_skill", setup=lambda: page.evaluate("window.__app.openSkill()"), wait=2500)
        page.keyboard.press("Escape")
        shot("08_about", setup=lambda: page.evaluate("window.__app.openAbout()"), wait=800)
        browser.close()
    httpd.shutdown()
    print(f"console errors/warnings: {len(errors)}")
    for e in errors[:20]:
        print("  " + e)


if __name__ == "__main__":
    main()
