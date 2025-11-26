import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from swag.posteriors.swag import SWAG
from swag.models import TwoDMLP  
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--out_path", type=str, default="./two_moons_decision_boundary.png")
parser.add_argument("--n_models", type=int, default=30)
parser.add_argument("--n_data", type=int, default=2000)
parser.add_argument("--noise", type=float, default=0.1)
parser.add_argument("--n_bins", type=int, default=10)  # kan bruges til ECE senere
args = parser.parse_args()

X, y = make_moons(n_samples=args.n_data, noise=args.noise)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y)

ckpt_path = args.model_path

model_cfg = TwoDMLP
model = SWAG(model_cfg.base, no_cov_mat=False, max_num_models=20, *model_cfg.args, **model_cfg.kwargs)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X, y = X.to(device), y.to(device)

x_min, x_max = X[:,0].min().cpu().item() - 0.5, X[:,0].max().cpu().item() + 0.5
y_min, y_max = X[:,1].min().cpu().item() - 0.5, X[:,1].max().cpu().item() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

plt.figure(figsize=(8,6))

for i in range(args.n_models):
    model.sample(scale=1.0)
    with torch.no_grad():
        logits = model(grid)
        probs = torch.softmax(logits, dim=1)[:,1]
    Z = probs.reshape(xx.shape).cpu().numpy()
    plt.contour(xx, yy, Z, levels=[0.5], colors='blue', alpha=0.2)

model.sample(scale=0.0)
with torch.no_grad():
    logits = model(grid)
    probs = torch.softmax(logits, dim=1)[:,1]
Z = probs.reshape(xx.shape).cpu().numpy()
plt.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=2)

plt.scatter(X[:,0].cpu(), X[:,1].cpu(), c=y.cpu(), cmap="coolwarm", edgecolor="k", s=20)
plt.title("SWAG Decision Boundaries with Uncertainty")
plt.xlabel("x1")
plt.ylabel("x2")

os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
plt.savefig(args.out_path)
print("Saved:", args.out_path)
