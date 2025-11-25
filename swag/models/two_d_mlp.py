import torch.nn as nn

__all__ = ["TwoDMLP"]   # modelnavnet (config-navnet)

class TwoDMLPBase(nn.Module):   # ← base-model, må IKKE hedde det samme som config
    def __init__(self, num_classes=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TwoDMLP:
    """Model config, så SWAG kan bygge modellen dynamisk"""
    base = TwoDMLPBase
    args = ()
    kwargs = {}
    transform_train = None
    transform_test = None
