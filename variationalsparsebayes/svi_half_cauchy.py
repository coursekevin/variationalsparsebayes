import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from typing import Union


class NormalMeanFieldVariational(nn.Module):
    """
    Mean field parameterization for normal variational distribution

    Args:
        mu_init (Tensor): mean initialization
        log_sigma_init (Tensor): log stdev initialization
    """

    def __init__(self, mu_init: Tensor, log_sigma_init: Tensor) -> None:
        super().__init__()
        # check that the input shapes are the same
        assert len(mu_init) == len(log_sigma_init), "init shapes must be equal."
        assert mu_init.shape == log_sigma_init.shape, "init shapes must be equal."
        self.d = len(mu_init)
        self.total_weights = len(mu_init)
        # saving parameters
        self.sparse_index = torch.ones(self.d).bool()

        self.mu = mu_init
        self.log_sigma = log_sigma_init

    @property
    def mu(self) -> Tensor:
        return self.__mu[self.sparse_index]

    @mu.setter
    def mu(self, value: Tensor):
        self.__mu = Parameter(value)

    @property
    def log_sigma(self) -> Tensor:
        return self.__log_sigma[self.sparse_index]

    @log_sigma.setter
    def log_sigma(self, value: Tensor):
        self.__log_sigma = Parameter(value)

    def var(self) -> Tensor:
        return torch.exp(self.log_sigma).pow(2)

    def update_sparse_index(self, sparse_index: Tensor) -> None:
        assert (
            len(sparse_index) == self.total_weights
        ), "Sparse index should be a bool array masking unimportant weights."
        self.sparse_index = sparse_index
        self.d = int(self.sparse_index.sum())

    def forward(self, n: int) -> Tensor:
        """
        Generates reparameterized samples

        Args:
            n (int): number of reparam samples

        Returns:
            Tensor: (n,d) reparameterized samples from variational distribution
        """
        sigma = torch.exp(self.log_sigma)
        return self.mu + torch.randn(n, self.d) * sigma


class LogNormalMeanFieldVariational(NormalMeanFieldVariational):
    """
    Mean field parameterization for log normal variational distribution

    Args:
        mu_init (Tensor): mean initialization
        log_sigma_init (Tensor): log stdev initialization
    """

    def __init__(self, mu_init: Tensor, log_sigma_init: Tensor) -> None:
        super().__init__(mu_init, log_sigma_init)

    def forward(self, n: int) -> Tensor:
        """
        Generates reparameterized samples

        Args:
            n (int): number of reparam samples

        Returns:
            Tensor: (n,d) reparameterized samples from variational distribution
        """
        return torch.exp(super().forward(n))


class SVIHalfCauchyPrior(nn.Module):
    """
    Class for performing sparse Bayesian learning using stochastic variational inference. This class provides 
    utilities for generating reparameterized samples from the variational distribution and computing the 
    KL-divergence between the variational distribution and the prior exactly     

    Args:
        d (int): number of parameters
        tau (Union[Tensor, float]): global half-cauchy scale parameter

    Attributes:
        tau (Tensor): global half-cauchy scale prior
        s_a (LogNormalMeanFieldVariational): log normal variational distribution for s_a
        s_a (LogNormalMeanFieldVariational): log normal variational distribution for s_b
        gamma_a (LogNormalMeanFieldVariational): log normal variational distribution for gamma_a
        gamma_b (LogNormalMeanFieldVariational): log normal variational distribution for gamma_b
        w_tilde (NormalMeanFieldVariational): normal variational distribution for w_tilde
        sparse_index (Tensor): index of sparse weights
        purning_tol (Union[Tensor, float]): pruning tolerance
    """

    def __init__(self, d: int, tau: Union[Tensor, float], w_init: Tensor = None):
        super().__init__()
        if isinstance(tau, float):
            tau = torch.tensor(tau)
        self.register_buffer("tau", tau)
        self.s_a = LogNormalMeanFieldVariational(
            torch.zeros(1), -6.0 + torch.randn(1) * 1e-4
        )
        self.s_b = LogNormalMeanFieldVariational(
            torch.zeros(1), -6.0 + torch.randn(1) * 1e-4
        )
        self.gamma_a = LogNormalMeanFieldVariational(
            torch.zeros(d), -6.0 + torch.randn(d) * 1e-4
        )
        self.gamma_b = LogNormalMeanFieldVariational(
            torch.zeros(d), -6.0 + torch.randn(d) * 1e-4
        )
        if w_init is None:
            w_init = torch.randn(d)
        self.w_tilde = NormalMeanFieldVariational(w_init, -6.0 + torch.randn(d) * 1e-4)
        self.register_buffer("sparse_index", torch.arange(d))
        self.pruning_tol = 0.0

    def _log_normal_reparam(
        self,
        n: int,
        d: int,
        mu_a: Tensor,
        mu_b: Tensor,
        log_sigma_a: Tensor,
        log_sigma_b: Tensor,
    ) -> Tensor:
        """
        Implements the reparameterization sampling trick described in Louizos 2017

        Args:
            n (int): number of reparam samples
            d (int): dimension of input
            mu_a (Tensor): log normal mean of r.v. a
            mu_b (Tensor): log normal mean of r.v. b
            log_sigma_a (Tensor): log stdev of r.v. a
            log_sigma_b (Tensor): log stdev of r.v. b

        Returns:
            Tensor: (n,d) reparmeterized samples 
        """
        mu = 0.5 * (mu_a + mu_b)
        var = 0.25 * (torch.exp(log_sigma_a).pow(2) + torch.exp(log_sigma_b).pow(2))
        return torch.exp(mu + torch.randn(n, d) * var.sqrt())

    def get_reparam_weights(self, n: int) -> Tensor:
        """
        Generate reparameterized samples 

        Args:
            n (int): number of reparam samples

        Returns:
            Tensor: weights
        """
        global_samples = self._log_normal_reparam(
            n,
            self.s_a.d,
            self.s_a.mu,
            self.s_b.mu,
            self.s_a.log_sigma,
            self.s_b.log_sigma,
        )
        local_samples = self._log_normal_reparam(
            n,
            self.gamma_a.d,
            self.gamma_a.mu,
            self.gamma_b.mu,
            self.gamma_a.log_sigma,
            self.gamma_b.log_sigma,
        )
        w_tilde_samples = self.w_tilde(n)
        return w_tilde_samples * local_samples * global_samples

    def _kl_s_a(self) -> Tensor:
        return -(
            torch.log(self.tau)
            - self.tau.pow(2) * torch.exp(self.s_a.mu + 0.5 * self.s_a.var())
            + 0.5 * (self.s_a.mu + 2 * self.s_a.log_sigma + 1 + math.log(2))
        )

    def _kl_s_b(self) -> Tensor:
        return -(
            -torch.exp(0.5 * self.s_b.var() - self.s_b.mu)
            + 0.5 * (-self.s_b.mu + 2 * self.s_b.log_sigma + 1 + math.log(2))
        )

    def _kl_gamma_a(self) -> Tensor:
        return -torch.sum(
            -torch.exp(self.gamma_a.mu + self.gamma_a.var() / 2)
            + 0.5 * (self.gamma_a.mu + 2 * self.gamma_a.log_sigma + 1 + math.log(2))
        )

    def _kl_gamma_b(self) -> Tensor:
        return -torch.sum(
            -torch.exp(-self.gamma_b.mu + self.gamma_b.var() / 2)
            + 0.5 * (-self.gamma_b.mu + 2 * self.gamma_b.log_sigma + 1 + math.log(2))
        )

    def _kl_w_tilde(self) -> Tensor:
        return -0.5 * torch.sum(
            1 + 2 * self.w_tilde.log_sigma - self.w_tilde.mu.pow(2) - self.w_tilde.var()
        )

    def kl_divergence(self) -> Tensor:
        """
        Computes the KL divergence for the approximating posteriors

        Returns:
            Tensor: kl divergence 
        """
        kl_sa = self._kl_s_a()
        kl_sb = self._kl_s_b()
        kl_gamma_a = self._kl_gamma_a()
        kl_gamma_b = self._kl_gamma_b()
        kl_w_tilde = self._kl_w_tilde()
        return kl_sa + kl_sb + kl_gamma_a + kl_gamma_b + kl_w_tilde

    def _compute_sparsity_tolerance(self, negative_log_mode: Tensor) -> Tensor:
        """
        Provides a reasonable pruning tolerance using the mid range of the
        negative log modes 

        Args:
            negative_log_mode (Tensor): negative log mode of the weight est

        Returns:
            Tensor: suggested pruning tolerance
        """
        return (negative_log_mode.max() + negative_log_mode.min()) / 2

    def update_sparse_index(self) -> Tensor:
        """
        Updates the sparse_index by pruning based on the negative log-mode 

        Returns: 
            Tensor: negative log mode for each parameter
        """
        mu_zi = 0.5 * (self.s_a.mu + self.s_b.mu + self.gamma_a.mu + self.gamma_b.mu)
        var_zi = 0.25 * (
            self.s_a.var() + self.s_b.var() + self.gamma_a.var() + self.gamma_b.var()
        )
        negative_log_mode = var_zi - mu_zi
        self.pruning_tol = self._compute_sparsity_tolerance(negative_log_mode)
        self.sparse_index = negative_log_mode <= self.pruning_tol
        self._propogate_sparse_index(self.sparse_index)
        return negative_log_mode

    def _propogate_sparse_index(self, sparse_index) -> None:
        """
        Propogates the sparsity inducing index to the variational distributions
        """
        self.gamma_a.update_sparse_index(sparse_index)
        self.gamma_b.update_sparse_index(sparse_index)
        self.w_tilde.update_sparse_index(sparse_index)


if __name__ == "__main__":
    print(2.0)
    svi = SVIHalfCauchyPrior(10, torch.tensor(1.0))
    print(svi.get_reparam_weights(20).shape)

