"""Compatibility loader exports backed by spyx.loaders.

These utilities are data-input helpers and remain JAX/NumPy based in upstream
spyx. We re-export them so spyx_mlx mirrors package surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPYX_SRC = _ROOT / "spyx" / "src"
if str(_SPYX_SRC) not in sys.path:
    sys.path.insert(0, str(_SPYX_SRC))

from spyx.loaders import *  # noqa: F401,F403
