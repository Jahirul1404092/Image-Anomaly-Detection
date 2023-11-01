"""Algorithms for decomposition and dimensionality reduction."""



from .pca import PCA
from .random_projection import SparseRandomProjection

__all__ = ["PCA", "SparseRandomProjection"]
