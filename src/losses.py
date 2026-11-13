"""
Project HQ-IDS-029: Class-Weighted Multi-Class Focal Loss
=========================================================
Implements Focal Loss with dynamic inverse-frequency class weighting
tailored for severe class imbalance (e.g. 532:1 ratio in UNSW-NB15).

Formulation:
    alpha_c = N_total / (K * N_c)
    p_t = Softmax(logits)_y
    L_FL = -alpha_y * (1 - p_t)^gamma * log(p_t + eps)
"""

from src import determinism as _determinism  # noqa: F401  (pins threads; see src/determinism.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassWeightedFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with class-specific alpha weighting:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes) raw model outputs before softmax
        targets: (B,) class indices in {0, ..., num_classes-1}
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        # Gather probabilities of true classes
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.clamp(p_t, self.eps, 1.0 - self.eps)

        # Compute focal modulation: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Log probability
        log_p_t = torch.log(p_t)

        # Loss per sample
        focal_loss = -modulating_factor * log_p_t

        # Apply class weights if available
        if self.class_weights is not None:
            alpha = self.class_weights[targets]
            focal_loss = alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    print("Testing ClassWeightedFocalLoss...")
    # 10 classes with severe imbalance
    counts = torch.tensor([1597., 545., 52., 56., 318., 987., 1112., 296., 32., 5.])
    total = counts.sum()
    weights = total / (10.0 * counts)
    weights = weights / weights.mean()

    criterion = ClassWeightedFocalLoss(class_weights=weights, gamma=2.0)
    dummy_logits = torch.randn(16, 10, requires_grad=True)
    dummy_targets = torch.randint(0, 10, (16,))
    loss = criterion(dummy_logits, dummy_targets)
    loss.backward()
    print("Loss value:", loss.item())
    print("Grad shape:", dummy_logits.grad.shape)
    assert dummy_logits.grad is not None, "Gradient computation failed"
    print("Self-test PASSED successfully!")
