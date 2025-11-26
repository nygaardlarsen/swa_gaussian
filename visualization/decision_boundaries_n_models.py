import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons
from swag.posteriors.swag import SWAG
from swag.models import TwoDMLP  
import os
import argparse
from torchmetrics.classification import BinaryCalibrationError

def compute_ece(probs, labels, n_bins=10):
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i+1])
        count = mask.sum().item()
        if count > 0:
            avg_conf = probs[mask].mean().item()
            avg_acc = labels[mask].float().mean().item()
            ece += (count / probs.size(0)) * abs(avg_conf - avg_acc)
    return ece

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--out_path", type=str, default="./two_moons_decision_boundary.png")
parser.add_argument("--n_models", type=int, default=30)
parser.add_argument("--n_data", type=int, default=2000)
parser.add_argument("--noise", type=float, default=0.1)
parser.add_argument("--n_bins", type=int, default=10)
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

ece_metric = BinaryCalibrationError(n_bins=args.n_bins, norm='l1').to(device)

x_min, x_max = X[:,0].min().cpu().item() - 0.5, X[:,0].max().cpu().item() + 0.5
y_min, y_max = X[:,1].min().cpu().item() - 0.5, X[:,1].max().cpu().item() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

plt.figure(figsize=(8,6))

all_probs = []
for i in range(args.n_models):
    model.sample(scale=1.0)
    with torch.no_grad():
        logits_grid = model(grid)
        probs_grid = torch.softmax(logits_grid, dim=1)[:,1]
    Z = probs_grid.reshape(xx.shape).cpu().numpy()
    plt.contour(xx, yy, Z, levels=[0.5], colors='blue', alpha=0.2, label='SWAG boundaries' if i == 0 else None)

    with torch.no_grad():
        logits_data = model(X)
        probs_data = torch.softmax(logits_data, dim=1)[:,1]
        all_probs.append(probs_data)

model.sample(scale=0.0)
with torch.no_grad():
    logits_grid = model(grid)
    probs_grid = torch.softmax(logits_grid, dim=1)[:,1]
Z = probs_grid.reshape(xx.shape).cpu().numpy()
plt.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=2, label='Mean / SWAG solution')

plt.scatter(X[:,0].cpu(), X[:,1].cpu(), c=y.cpu(), cmap="coolwarm", edgecolor="k", s=20)

all_probs_tensor = torch.stack(all_probs)
mean_probs = all_probs_tensor.mean(dim=0)
my_ece_func = compute_ece(mean_probs, y, n_bins=args.n_bins)
torchmetric_ece = ece_metric(mean_probs, y).item()

plt.title("SWAG Decision Boundaries with Uncertainty")
plt.xlabel("x1")
plt.ylabel("x2")
plt.text(x_min + 0.1, y_max - 0.2, f"Torchmetric ECE = {torchmetric_ece:.4f}", color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.text(x_min + 0.1, y_max - 0.5, f"My ECE func= {my_ece_func:.4f}", color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, alpha=0.2, label='SWAG decision boundaries'),
    Line2D([0], [0], color='red', lw=2, label='Mean / SWAG solution')
]

plt.figtext(0.1, 0.01, f"Noise = {args.noise:.2f}, n_samples = {args.n_data}", fontsize=10, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7))


plt.legend(handles=legend_elements, loc='upper right')

os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
plt.savefig(args.out_path)
print("Saved:", args.out_path)
print(f"Torchmetric ECE: {torchmetric_ece:.4f}")
print(f"My own ECE: {my_ece_func:.4f}")
