from .matrix import FisherMatrixPosteriorEstimator
from .sampler import Laplace

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["Laplace", "FisherMatrixPosteriorEstimator", "__version__"]
