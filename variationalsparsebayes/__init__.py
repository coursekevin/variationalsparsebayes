from .sparse_glm import (
    SparsePolynomialFeatures,
    SparseGLMGaussianLikelihood,
    Polynomial,
)

from .svi_half_cauchy import (
    NormalMeanFieldVariational,
    LogNormalMeanFieldVariational,
    SVIHalfCauchyPrior,
)

from .sparse_bnn import SparseBNN, BayesianLinear, BayesianResidual, Identity

__version__ = "0.0.2"
