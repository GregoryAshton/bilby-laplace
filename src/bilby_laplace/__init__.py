from .matrix import FisherMatrixPosteriorEstimator
from .sampler import Fisher

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["Fisher", "FisherMatrixPosteriorEstimator", "__version__"]
