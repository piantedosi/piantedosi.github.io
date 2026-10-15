"""Build the browser package from the dag-explorer checkout.

The pygame game lives in that repository (see source.path). This script copies
dag.py and dag_explorer.py, packages them with pygbag, and replaces the site assets.

From the repository root:

    uv run --with-requirements games/causal_dag_discovery_game/requirements.txt games/causal_dag_discovery_game/build.py
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parents[1]
WEB_BUILD = GAME_DIR / "build" / "web"
SITE_DIR = REPO_ROOT / "assets" / "causal_dag_discovery_game"
SOURCE_FILE = GAME_DIR / "source.path"
WIDTH = 640
HEIGHT = 360
REQUIRED = ("dag.py", "dag_explorer.py")


def configured_path() -> str:
    env = os.environ.get("DAG_EXPLORER", "").strip()
    if env:
        return env
    if not SOURCE_FILE.is_file():
        raise SystemExit(f"set DAG_EXPLORER or write the checkout path in {SOURCE_FILE.name}")
    for line in SOURCE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise SystemExit(f"{SOURCE_FILE.name} has no checkout path")


def has_game(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED)


def windows_path_for_wsl(linux_path: str) -> Path | None:
    try:
        result = subprocess.run(
            ["wsl", "wslpath", "-w", linux_path],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    converted = result.stdout.strip()
    if not converted:
        return None
    return Path(converted)


def resolve_source() -> Path:
    raw = configured_path()
    direct = Path(raw)
    if has_game(direct):
        return direct
    if platform.system() == "Windows" and raw.startswith("/"):
        converted = windows_path_for_wsl(raw)
        if converted is not None and has_game(converted):
            return converted
    raise SystemExit(f"dag-explorer checkout not found at {raw} (need dag.py and dag_explorer.py)")


def stage_sources(source: Path) -> None:
    entry = (source / "dag_explorer.py").read_text(encoding="utf-8")
    if "await asyncio.sleep" not in entry:
        raise SystemExit(
            "dag_explorer.py must await asyncio.sleep(0) in the game loop so the browser build can paint"
        )
    shutil.copy2(source / "dag.py", GAME_DIR / "dag.py")
    (GAME_DIR / "main.py").write_text(entry, encoding="utf-8")


def publish() -> None:
    if not (WEB_BUILD / "index.html").is_file():
        raise SystemExit(f"pygbag did not write {WEB_BUILD / 'index.html'}")

    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    shutil.copytree(WEB_BUILD, SITE_DIR)

    index = SITE_DIR / "index.html"
    text = index.read_text(encoding="utf-8")
    patched, count = re.subn(r"can_close\s*:\s*\d+", "can_close : 1", text, count=1)
    if count != 1:
        raise SystemExit("pygbag index.html has no can_close setting to keep the site links clickable")
    index.write_text(patched, encoding="utf-8")


def main() -> None:
    source = resolve_source()
    print(f"using {source}")
    stage_sources(source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pygbag",
            "--build",
            "--width",
            str(WIDTH),
            "--height",
            str(HEIGHT),
            str(GAME_DIR),
        ],
        check=True,
        cwd=GAME_DIR,
    )
    publish()
    print(f"wrote {SITE_DIR}")


if __name__ == "__main__":
    main()
