import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import optimizer
from variationalsparsebayes import *
import numpy as np
import math

plt.style.use("default")

torch.set_default_dtype(torch.float64)

plt.style.use("default")

n_data = 1000
train_x = torch.linspace(0, 1, 1000)
noise = 0.2
train_y = torch.stack(
    [
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * noise,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * noise,
        torch.sin(train_x * (2 * math.pi))
        + 2 * torch.cos(train_x * (2 * math.pi))
        + torch.randn(train_x.size()) * noise,
        -torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * noise,
    ],
    -1,
)
n_tasks = train_y.size(-1)


def sampler():
    idx = np.random.choice(n_data, size=(64,), replace=False)
    return (train_x[idx].unsqueeze(-1), train_y[idx, :])


if __name__ == "__main__":
    d = 1
    model = SparseBNN(
        in_features=1, out_features=n_tasks, n_hidden=30, n_layers=2, n_reparam=16
    )
    num_epochs = 1000
    beta_warmup_iters = 250
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    fig, axs = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 3))
    beta = 0.0
    for j in range(1, num_epochs + 1):
        optimizer.zero_grad()
        x_batch, y_batch = sampler()
        beta = min(1.0, (1.0 * j) / (beta_warmup_iters))
        model.reparam_sample()
        loss = -model.elbo(x_batch, y_batch, n_data, beta=beta)
        loss.backward()
        optimizer.step()

        if j % 100 == 0:
            print(f"iter: {j:05d} | loss: {loss.item():.2f}")
            # plotting
            x_t = torch.linspace(0, 1, 300)
            y_pred = model(x_t.unsqueeze(-1))
            mu = y_pred.mean(0)
            var = y_pred.var(0) + model.sigma.pow(2)
            lb = mu - 2 * var.sqrt()
            ub = mu + 2 * var.sqrt()
            for j, ax in enumerate(axs):
                ax.cla()
                ax.plot(train_x, train_y[:, j], "C7o", alpha=0.3)
                with torch.no_grad():
                    ax.plot(x_t, mu[:, j], "C0")
                    ax.fill_between(x_t, lb[:, j], ub[:, j], color="C0", alpha=0.3)
            fig.tight_layout()
            plt.pause(1 / 60)
    plt.show()
