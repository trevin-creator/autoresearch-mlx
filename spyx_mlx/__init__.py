"""
spyx_mlx — MLX-native spiking neural network library.
API mirrors spyx but runs natively on Apple Silicon via MLX.
"""

from ._version import __version__
from . import axn, data, fn, nn

try:
	from . import experimental, loaders, nir
except Exception:  # optional JAX-backed compatibility modules
	experimental = None
	loaders = None
	nir = None

__all__ = [
	"__version__",
	"axn",
	"data",
	"fn",
	"nn",
	"experimental",
	"loaders",
	"nir",
]
