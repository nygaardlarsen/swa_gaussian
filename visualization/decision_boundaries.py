import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from swag.posteriors.swag import SWAG
from swag.models import TwoDMLP   # du skal have denne i models/
import os

X, y = make_moons(n_samples=2000, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y)


model_dir = "/work/SWAG/models/2dmlp_two_moons_swag"
ckpt_path = os.path.join(model_dir, "swag-100.pt")  # eller swa.pkl afhængigt af dit run

print("Loading:", ckpt_path)

model_cfg = TwoDMLP              # config-objekt
model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
model.load_state_dict(torch.load(ckpt_path)["model_state"])
model.eval()

x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_torch = torch.tensor(grid, dtype=torch.float32)


with torch.no_grad():
    logits = model(grid_torch)
    probs = torch.softmax(logits, dim=1)[:,1]   # probability for class 1
    Z = probs.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.7)

plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolor="k", s=20)

plt.title("Decision Boundary (Two Moons)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
