# variationalsparsebayes

This library provides a PyTorch implementation for learning sparse models with with half-cauchy priors using stochastic variational inference.

# Features

The main features of the library are methods for performing:

- [sparse polynomial regression](https://github.com/coursekevin/variationalsparsebayes/blob/main/examples/sparse_poly_regression.py)
- sparse learning with [precomputed features](https://github.com/coursekevin/variationalsparsebayes/blob/main/examples/support_vectors.py)
- sparse learning of [Bayesian neural networks](https://github.com/coursekevin/variationalsparsebayes/blob/main/examples/sparse_bnn_regression.py).

To implement your own custom features, you can inherit from the [SparseFeaturesLibrary](https://github.com/coursekevin/variationalsparsebayes/blob/main/variationalsparsebayes/sparse_glm.py) class.

More generally you can use the [SVIHalfCauchyPrior](https://github.com/coursekevin/variationalsparsebayes/blob/main/variationalsparsebayes/svi_half_cauchy.py) class to perform sparse regression with _any_ parameterized model. To do so you need to define a method which takes reparameterized sample weights and computes the expected log-likelihood of your data using these weights.
