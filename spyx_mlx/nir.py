"""Compatibility NIR exports backed by spyx.nir.

NIR conversion logic is framework-bridging and re-exported for API parity.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPYX_SRC = _ROOT / "spyx" / "src"
if str(_SPYX_SRC) not in sys.path:
    sys.path.insert(0, str(_SPYX_SRC))

from spyx.nir import *  # noqa: F401,F403
