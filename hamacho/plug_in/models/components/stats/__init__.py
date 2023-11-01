"""Statistical functions."""



from .kde import GaussianKDE
from .multi_variate_gaussian import MultiVariateGaussian

__all__ = ["GaussianKDE", "MultiVariateGaussian"]
