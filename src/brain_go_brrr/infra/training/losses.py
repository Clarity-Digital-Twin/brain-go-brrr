"""Custom loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """Initialize Focal Loss.
        
        Args:
            alpha: Per-class weights (shape: [num_classes])
            gamma: Focusing parameter (typically 2.0)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions (shape: [batch, num_classes])
            targets: Ground truth labels (shape: [batch])
        
        Returns:
            Focal loss value
        """
        # Get class probabilities
        p = F.softmax(inputs, dim=-1)
        
        # Get class log probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Get probability of true class
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        loss = focal_term * ce_loss
        
        # Apply per-class weights if provided
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy with class weights support."""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """Initialize weighted label smoothing loss.
        
        Args:
            smoothing: Label smoothing parameter (0.0 = no smoothing)
            weight: Per-class weights
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted label smoothing loss.
        
        Args:
            inputs: Model predictions (shape: [batch, num_classes])
            targets: Ground truth labels (shape: [batch])
        
        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Compute KL divergence loss
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        # Apply class weights if provided
        if self.weight is not None:
            if self.weight.device != loss.device:
                self.weight = self.weight.to(loss.device)
            weight_expanded = self.weight[targets]
            loss = loss * weight_expanded
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss