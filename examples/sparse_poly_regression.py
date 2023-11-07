import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import optimizer
from variationalsparsebayes import *
import numpy as np

torch.set_default_dtype(torch.float64)

plt.style.use("default")

n_data = 1000
x = torch.linspace(-3, 3, n_data)
noise = 1e-1
y = -0.1 * x + 2.0 * x.pow(3)
y_data = y + torch.randn_like(x) * noise


def sampler():
    idx = np.random.choice(np.arange(n_data), 128, replace=False)
    return (x[idx].unsqueeze(-1), y_data[idx])


if __name__ == "__main__":
    d = 1
    features = SparsePolynomialFeatures(d, 8, include_bias=True, input_labels=["x"])
    model = SparseGLMGaussianLikelihood(d, features, noise=noise, learn_noise=False)
    opt_summary = model.optimize(
        data_sampler=sampler, n_data_total=n_data, print_progress=True
    )
    # prune basis
    model.prune_basis()
    print(model.features)
    # plotting
    x_t = x  # torch.linspace(-2, 2, 500)
    mu, cov = model(x_t.unsqueeze(-1))
    lb = mu - 2 * torch.sqrt(cov.diag())
    ub = mu + 2 * torch.sqrt(cov.diag())

    plt.plot(x, y_data, "C7o", alpha=0.3)
    with torch.no_grad():
        plt.plot(x_t, mu, "C0")
        plt.fill_between(x_t, lb, ub, color="C0", alpha=0.3)
    plt.plot(x, y, "C7--", alpha=1.0)
    plt.show()
