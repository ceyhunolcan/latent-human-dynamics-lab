"""Launch the Streamlit dashboard.

Thin wrapper around ``streamlit run`` so contributors don't have to remember
the exact module path. Run from repo root::

    python scripts/launch_dashboard.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "src" / "dashboard" / "app.py"


def main() -> int:
    if shutil.which("streamlit") is None:
        print(
            "Streamlit is not installed. Run `pip install -r requirements.txt` "
            "and try again.",
            file=sys.stderr,
        )
        return 1
    cmd = ["streamlit", "run", str(APP)] + sys.argv[1:]
    print(f"Launching: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
