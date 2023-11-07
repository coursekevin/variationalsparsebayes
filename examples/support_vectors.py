from variationalsparsebayes.sparse_glm import SparsePrecomputedFeatures

# import sparsebayes
import matplotlib.pyplot as plt
import torch
import numpy as np
from variationalsparsebayes import *
import math

import time

torch.set_default_dtype(torch.float64)
torch.manual_seed(2021)


plt.style.use("default")

# n_data = 1000
n_data = 100
x = torch.linspace(-10, 10, n_data)

noise = 1e-1  # 0.2
y = torch.sinc(x / math.pi)
# y_data = y + (torch.rand(n_data) * (0.2 + 0.2) - 0.2)
y_data = y + torch.randn(n_data) * noise


def rbf(x_in):
    x_in = x_in.unsqueeze(-1)
    xn = x.unsqueeze(0)
    d = (x_in - xn).pow(2)
    l = 3.0
    return torch.exp(-d / (l**2))


phi = rbf(x)


def data_sampler():
    return (phi, y_data)


if __name__ == "__main__":
    x_test = torch.linspace(-10, 10, n_data * 2)
    phi_test = rbf(x_test)
    features = SparsePrecomputedFeatures(n_data)
    start_time = time.time()
    model = SparseGLMGaussianLikelihood(
        n_data, features, noise=noise, learn_noise=False, tau=1.0
    )
    opt_summary = model.optimize(
        data_sampler=data_sampler,
        n_data_total=n_data,
        lr=1e-1,
        beta_warmup_iters=3000,
        max_iter=20000,
        n_reparams=20,
        print_progress=True,
    )
    print("Time elapsed: {}".format(time.time() - start_time))
    # prune basis
    model.prune_basis()
    axs = [0]
    f, axs[0] = plt.subplots(1, 1, sharey=True)
    mu, cov = model(phi_test)
    with torch.no_grad():
        axs[0].plot(x, y_data, "k.", alpha=0.3)
        axs[0].plot(
            x_test,
            mu,
            "--",
            label="half-cauchy, {} support vecs.".format(model.num_sparse_features),
        )
        lb = mu - 2 * torch.sqrt(cov.diag())
        ub = mu + 2 * torch.sqrt(cov.diag())
        axs[0].fill_between(x_test, lb, ub, alpha=0.3)
        axs[0].axis("off")
        axs[0].legend()
        axs[0].plot(
            x[model.sparse_index],
            y_data[model.sparse_index],
            "ro",
            fillstyle="none",
            linewidth=1.0,
            markersize=10,
        )
        axs[0].plot(x, y, "k-")
    plt.show()
