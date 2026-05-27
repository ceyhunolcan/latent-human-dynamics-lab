"""Repository-relative path resolution.

Anchors all paths to the repository root so that scripts can be run from
any working directory without sprinkling ``os.path`` calls throughout the
codebase.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at:  <repo>/src/utils/paths.py
# So the repository root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RAW_DIR: Path = DATA_DIR / "raw"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
CONFIGS_DIR: Path = REPO_ROOT / "configs"


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` if it doesn't exist, return the resolved Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cohort_csv() -> Path:
    """Default location of the synthetic cohort CSV."""
    return SYNTHETIC_DIR / "synthetic_cohort.csv"


def processed_csv() -> Path:
    """Default location of the engineered-features CSV."""
    return PROCESSED_DIR / "processed_features.csv"
