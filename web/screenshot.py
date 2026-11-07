"""Dev tool: screenshot every dashboard view with headless Chromium (Playwright).

    python -m web.screenshot            # writes web/screenshots/*.png (git-ignored)

Starts the real server (uvicorn, 127.0.0.1 only) on a free port with a TEMPORARY user database and a
throw-away admin account, signs in through the login form, steps through the views and reports any
browser console errors (including Content-Security-Policy violations). The browser lives in the
project's .playwright/ folder. Nothing touches the real data/auth/users.db.
"""
import os
import secrets
import socket
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright"))

from playwright.sync_api import sync_playwright  # noqa: E402

import uvicorn  # noqa: E402

from web.auth import AuthStore  # noqa: E402
from web.server import create_app  # noqa: E402
from web.settings import Settings  # noqa: E402

OUT = ROOT / "web" / "screenshots"
VIEWPORT = {"width": 1440, "height": 900}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(page, extra_ms=1800, timeout_s=60):
    # Poll from Python: page.wait_for_function would need eval(), which our Content-Security-Policy blocks.
    end = time.time() + timeout_s
    while time.time() < end:
        if page.evaluate("Boolean(window.__app) && document.body.dataset.ready === '1'"):
            break
        page.wait_for_timeout(200)
    else:
        raise TimeoutError("dashboard did not report ready")
    page.wait_for_timeout(extra_ms)  # basemap tiles and the image fade


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    db = Path(tmp.name) / "users.db"
    store = AuthStore(db)
    password = secrets.token_urlsafe(16)
    store.add_user("demo.admin", password, "admin")
    server = uvicorn.Server(uvicorn.Config(create_app(Settings("127.0.0.1", port, db, False), store),
                                           host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
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

        page.goto(f"{base}login")
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "00a_login.png"))
        print("  saved 00a_login.png")
        page.fill("#username", "demo.admin")
        page.fill("#password", "not the password")
        page.click("button[type=submit]")
        page.wait_for_url("**/login?error=invalid")
        page.screenshot(path=str(OUT / "00b_login_error.png"))
        print("  saved 00b_login_error.png")
        page.fill("#username", "demo.admin")
        page.fill("#password", password)
        page.click("button[type=submit]")
        page.wait_for_url(base)
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
        page.keyboard.press("Escape")
        shot("09_users_admin", setup=lambda: page.evaluate("window.__app.openUsers()"), wait=1000)
        page.keyboard.press("Escape")
        shot("10_why_overlay", setup=lambda: page.evaluate("window.__app.openWhy()"), wait=1500)
        page.keyboard.press("Escape")
        shot("11_drawer_blight_why", "site=nakuru&date=2025-04-15&h=2&layer=blight_hutton&view=forecast&mode=smooth")
        page.evaluate("window.__app.selectPixel(Math.floor(window.__app.state.pixels.nakuru.ids.length * 0.4))")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "11_drawer_blight_why.png"))
        browser.close()
    server.should_exit = True
    store.close()
    tmp.cleanup()
    print(f"console errors/warnings: {len(errors)}")
    for e in errors[:20]:
        print("  " + e)


if __name__ == "__main__":
    main()
