"""Front-end checks that need no browser: the JavaScript unit test for the coordinate parser (Node) and
static checks that the new pages and assets are served and that every module parses.
Skipped when Node.js is not installed."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
def test_coordinate_parser(tmp_path):
    copy = tmp_path / "coords.mjs"
    copy.write_text((STATIC / "coords.js").read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "coords.test.mjs"), copy.as_uri()],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr or r.stdout
    assert "checks passed" in r.stdout


@pytest.mark.skipif(NODE is None, reason="Node.js not installed")
@pytest.mark.parametrize("name", sorted(p.name for p in STATIC.glob("*.js")))
def test_every_module_parses(name, tmp_path):
    # copy as .mjs so Node parses it as an ES module without running it
    f = tmp_path / (Path(name).stem + ".mjs")
    f.write_text((STATIC / name).read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_pages_reference_only_existing_local_files():
    import re
    for page in STATIC.glob("*.html"):
        html = page.read_text(encoding="utf-8")
        for ref in re.findall(r'(?:src|href)="/([^"#?]+\.[a-z]+)"', html):
            assert (STATIC / ref).is_file(), f"{page.name} -> /{ref}"
