"""Real-data adapter layer.

Each adapter accepts a path (or directory) on disk, parses the source-specific
schema, and returns a DataFrame coerced to the canonical column schema defined
in :mod:`src.data.validation`. None of the adapters in this repository ship
with real data — they are runnable shells that document expected formats and
raise informative ``FileNotFoundError`` exceptions when called without data.

The adapters are deliberately thin. They are not opinionated about cleaning;
that is the job of :mod:`src.data.preprocessing` after the cohort is loaded.
"""

from .studentlife_adapter import load_studentlife
from .wesad_adapter import load_wesad
from .apple_health_adapter import load_apple_health
from .garmin_adapter import load_garmin
from .weather_adapter import load_weather

__all__ = [
    "load_studentlife",
    "load_wesad",
    "load_apple_health",
    "load_garmin",
    "load_weather",
]
