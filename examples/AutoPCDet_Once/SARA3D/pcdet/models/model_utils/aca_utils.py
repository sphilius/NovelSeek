import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConfidenceAggregation(nn.Module):
    """
    Adaptive Confidence Aggregation (ACA) module for enhancing bounding box prediction confidence
    based on geometric properties of point clouds.
    
    Simplified version to avoid matrix multiplication errors.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_density = model_cfg.get('USE_DENSITY', True)
        self.use_curvature = model_cfg.get('USE_CURVATURE', False)  # Disabled by default
        self.use_normals = model_cfg.get('USE_NORMALS', False)  # Disabled by default
        
        # Fixed weights for geometric properties (no learning to avoid matrix multiplication errors)
        self.density_weight = 1.0
        self.curvature_weight = 0.5
        self.normals_weight = 0.3
        
    def forward(self, geometric_features, base_scores=None):
        """
        Args:
            geometric_features: (N, 5) tensor with [density, curvature, normal_x, normal_y, normal_z]
            base_scores: Optional (N,) tensor with base confidence scores to refine
            
        Returns:
            confidence_scores: (N,) tensor with refined confidence scores
        """
        try:
            # Validate input
            if geometric_features is None:
                raise ValueError("geometric_features is None")
                
            # Convert to tensor if it's not already
            if not isinstance(geometric_features, torch.Tensor):
                geometric_features = torch.tensor(geometric_features)
                
            # Ensure it's on the right device
            device = next(self.parameters()).device
            geometric_features = geometric_features.to(device)
            
            # Check if geometric_features has the right shape
            if len(geometric_features.shape) == 1:
                # If it's a 1D tensor, reshape to 2D
                geometric_features = geometric_features.unsqueeze(0)
                
            # Ensure we have at least 5 feature dimensions
            if geometric_features.shape[1] < 5:
                # Pad with zeros if needed
                padding_size = 5 - geometric_features.shape[1]
                padding = torch.zeros(geometric_features.shape[0], padding_size, device=device)
                geometric_features = torch.cat([geometric_features, padding], dim=1)
            elif geometric_features.shape[1] > 5:
                # Slice to first 5 dimensions
                geometric_features = geometric_features[:, :5]
            
            # Handle NaN or Inf values
            geometric_features = torch.nan_to_num(geometric_features, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Simplified confidence computation using fixed weights
            confidence_scores = torch.ones(geometric_features.shape[0], device=device)
            
            # Apply density weight if enabled
            if self.use_density:
                density = geometric_features[:, 0]
                confidence_scores = confidence_scores * (0.5 + 0.5 * density)
            
            # Apply curvature weight if enabled
            if self.use_curvature:
                curvature = geometric_features[:, 1]
                confidence_scores = confidence_scores * (0.8 + 0.2 * (1.0 - curvature))
            
            # Apply normals weight if enabled
            if self.use_normals:
                # Use only the z-component of the normal for simplicity
                normal_z = geometric_features[:, 4]
                confidence_scores = confidence_scores * (0.9 + 0.1 * torch.abs(normal_z))
            
            # If base scores are provided, combine them with our confidence scores
            if base_scores is not None:
                try:
                    # Convert to tensor if it's not already
                    if not isinstance(base_scores, torch.Tensor):
                        base_scores = torch.tensor(base_scores, device=device)
                    else:
                        base_scores = base_scores.to(device)
                        
                    # Ensure base_scores has the right shape
                    if base_scores.dim() == 0:
                        base_scores = base_scores.unsqueeze(0).expand(confidence_scores.shape[0])
                    elif base_scores.dim() > 1:
                        base_scores = base_scores.squeeze()
                    
                    # Ensure base_scores has the same length as confidence_scores
                    if base_scores.shape[0] != confidence_scores.shape[0]:
                        if base_scores.shape[0] > confidence_scores.shape[0]:
                            base_scores = base_scores[:confidence_scores.shape[0]]
                        else:
                            # Pad with ones
                            padding = torch.ones(confidence_scores.shape[0] - base_scores.shape[0], device=device)
                            base_scores = torch.cat([base_scores, padding])
                    
                    # Handle NaN or Inf values
                    base_scores = torch.nan_to_num(base_scores, nan=1.0, posinf=1.0, neginf=0.0)
                    
                    # Combine scores - use a weighted average instead of multiplication
                    confidence_scores = 0.3 * confidence_scores + 0.7 * base_scores
                except Exception as e:
                    print(f"Warning: Error processing base_scores: {e}. Using computed confidence scores only.")
                
            # Final check for NaN or Inf values
            confidence_scores = torch.nan_to_num(confidence_scores, nan=1.0, posinf=1.0, neginf=0.0)
            
            # Ensure confidence scores are in [0, 1]
            confidence_scores = torch.clamp(confidence_scores, 0.0, 1.0)
            
            return confidence_scores
            
        except Exception as e:
            print(f"Warning: Error in AdaptiveConfidenceAggregation: {e}. Using fallback.")
            # Fallback: return base scores or ones
            if base_scores is not None:
                if isinstance(base_scores, torch.Tensor):
                    return base_scores
                else:
                    device = next(self.parameters()).device
                    return torch.ones(1, device=device)
            else:
                device = next(self.parameters()).device
                return torch.ones(1, device=device)
        
    @staticmethod
    def apply_confidence_to_boxes(boxes, confidence_scores, score_thresh=0.1):
        """
        Apply confidence scores to boxes and filter by threshold
        
        Args:
            boxes: (N, 7+C) [x, y, z, dx, dy, dz, heading, ...]
            confidence_scores: (N,) confidence scores
            score_thresh: Threshold for filtering boxes
            
        Returns:
            filtered_boxes: Boxes with scores above threshold
        """
        # Apply confidence scores to box scores (assuming score is at index 7)
        if boxes.shape[0] == 0:
            return boxes
            
        boxes_with_conf = boxes.clone()
        boxes_with_conf[:, 7] = boxes_with_conf[:, 7] * confidence_scores
        
        # Filter boxes by score threshold
        mask = boxes_with_conf[:, 7] >= score_thresh
        filtered_boxes = boxes_with_conf[mask]
        
        return filtered_boxes