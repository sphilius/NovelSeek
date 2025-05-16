import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy


class AdaptiveAugmentation:
    """
    Implements adaptive data-driven augmentation for HARCNet.
    Dynamically adjusts geometric and MixUp augmentations based on data distribution.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0):
        """
        Args:
            alpha: Weight for variance component in geometric augmentation
            beta: Weight for entropy component in geometric augmentation
            gamma: Scaling factor for MixUp interpolation
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_variance(self, x):
        """Compute variance across feature dimensions"""
        # x shape: [B, C, H, W]
        # Compute variance across channels for each spatial location
        var = torch.var(x, dim=1, keepdim=True)  # [B, 1, H, W]
        return var.mean(dim=[1, 2, 3])  # [B]
    
    def compute_entropy(self, probs):
        """Compute entropy of probability distributions"""
        # probs shape: [B, C] where C is number of classes
        # Ensure valid probability distribution
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        log_probs = torch.log(probs)
        entropy_val = -torch.sum(probs * log_probs, dim=1)  # [B]
        return entropy_val
    
    def get_geometric_strength(self, x, model=None, probs=None):
        """
        Compute geometric augmentation strength based on sample variance and entropy
        S_g(x_i) = α·Var(x_i) + β·Entropy(x_i)
        """
        var = self.compute_variance(x)
        
        # If model predictions are provided, use them for entropy calculation
        if probs is None and model is not None:
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)
        
        if probs is not None:
            ent = self.compute_entropy(probs)
        else:
            # Default entropy if no predictions available
            ent = torch.ones_like(var)
            
        # Normalize to [0, 1] range
        var = (var - var.min()) / (var.max() - var.min() + 1e-8)
        ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
        
        strength = self.alpha * var + self.beta * ent
        return strength
    
    def get_mixup_params(self, y, num_classes=100):
        """
        Generate MixUp parameters based on label entropy
        λ ~ Beta(γ·Entropy(y), γ·Entropy(y))
        """
        # Convert labels to one-hot encoding
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        
        # Compute entropy of ground truth labels (across batch)
        batch_entropy = self.compute_entropy(y_onehot.mean(dim=0, keepdim=True)).item()
        
        # Generate mixup coefficient from Beta distribution
        alpha = self.gamma * batch_entropy
        alpha = max(0.1, min(alpha, 2.0))  # Bound alpha between 0.1 and 2.0
        
        lam = np.random.beta(alpha, alpha)
        
        # Generate random permutation for mixing
        batch_size = y.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        return lam, index
    
    def apply_mixup(self, x, y, num_classes=100):
        """Apply MixUp augmentation with adaptive coefficient"""
        lam, index = self.get_mixup_params(y, num_classes)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class TemporalConsistencyRegularization:
    """
    Implements decayed temporal consistency regularization for HARCNet.
    Reduces noise in pseudo-labels by incorporating past predictions.
    """
    def __init__(self, memory_size=5, decay_rate=2.0, consistency_weight=0.1):
        """
        Args:
            memory_size: Number of past predictions to store (K)
            decay_rate: Controls the decay of weights for past predictions (τ)
            consistency_weight: Weight for consistency loss (λ_consistency)
        """
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.consistency_weight = consistency_weight
        self.prediction_history = {}  # Store past predictions for each sample
        
    def compute_decay_weights(self):
        """
        Compute exponentially decaying weights
        ω_k = e^(-k/τ) / Σ(e^(-k/τ))
        """
        weights = torch.exp(-torch.arange(1, self.memory_size + 1) / self.decay_rate)
        return weights / weights.sum()
    
    def update_history(self, indices, predictions):
        """Update prediction history for each sample"""
        for i, idx in enumerate(indices):
            idx = idx.item()
            if idx not in self.prediction_history:
                self.prediction_history[idx] = []
            
            # Add current prediction to history
            self.prediction_history[idx].append(predictions[i].detach())
            
            # Keep only the most recent K predictions
            if len(self.prediction_history[idx]) > self.memory_size:
                self.prediction_history[idx].pop(0)
    
    def get_aggregated_predictions(self, indices):
        """
        Get aggregated predictions for each sample using decay weights
        ỹ_i = Σ(ω_k · ŷ_i^(t-k))
        """
        weights = self.compute_decay_weights().to(indices.device)
        aggregated_preds = []
        
        for i, idx in enumerate(indices):
            idx = idx.item()
            if idx in self.prediction_history and len(self.prediction_history[idx]) > 0:
                # Get available history (might be less than memory_size)
                history = self.prediction_history[idx]
                history_len = len(history)
                
                if history_len > 0:
                    # Use available weights
                    available_weights = weights[-history_len:]
                    available_weights = available_weights / available_weights.sum()
                    
                    # Compute weighted sum
                    weighted_sum = torch.zeros_like(history[0])
                    for j, pred in enumerate(history):
                        weighted_sum += available_weights[j] * pred
                    
                    aggregated_preds.append(weighted_sum)
                else:
                    # No history available, use zeros
                    aggregated_preds.append(torch.zeros_like(history[0]))
            else:
                # No history for this sample, return None
                aggregated_preds.append(None)
        
        return aggregated_preds
    
    def compute_consistency_loss(self, current_preds, indices):
        """
        Compute consistency loss between current and aggregated past predictions
        L_consistency(x_i) = ||ŷ_i^(t) - Σ(ω_k · ŷ_i^(t-k))||^2_2
        """
        aggregated_preds = self.get_aggregated_predictions(indices)
        loss = 0.0
        valid_samples = 0
        
        for i, agg_pred in enumerate(aggregated_preds):
            if agg_pred is not None:
                # Compute MSE between current and aggregated predictions
                sample_loss = F.mse_loss(current_preds[i], agg_pred)
                loss += sample_loss
                valid_samples += 1
        
        # Return average loss if there are valid samples
        if valid_samples > 0:
            return loss / valid_samples
        else:
            # Return zero loss if no valid samples
            return torch.tensor(0.0).to(current_preds.device)
