import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from swag.posteriors.swag import SWAG
from swag.models import TwoDMLP
import os

X, y = make_moons(n_samples=2000, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y)

model_dir = "/work/SWAG/models/2dmlp_two_moons_swag"
ckpt_path = os.path.join(model_dir, "swag-200.pt")

model_cfg = TwoDMLP
model = SWAG(model_cfg.base, no_cov_mat=False, max_num_models=20, *model_cfg.args, **model_cfg.kwargs)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.sample(scale=0.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)
y = y.to(device)

x_min, x_max = X[:, 0].min().cpu().item() - 0.5, X[:, 0].max().cpu().item() + 0.5
y_min, y_max = X[:, 1].min().cpu().item() - 0.5, X[:, 1].max().cpu().item() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_torch = torch.tensor(grid, dtype=torch.float32).to(device)

with torch.no_grad():
    logits = model(grid_torch)
    probs = torch.softmax(logits, dim=1)[:, 1]
    Z = probs.reshape(xx.shape).cpu().numpy()

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.7)
plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), cmap="coolwarm", edgecolor="k", s=20)
plt.title("Decision Boundary (Two Moons)")
plt.xlabel("x1")
plt.ylabel("x2")


out_dir = "/work/SWAG/visualization"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "two_moons_decision_boundary.png")
plt.savefig(out_path)
print("Saved:", out_path)
