import torch
import torch.nn.functional as F

def masked_cross_entropy(pred: torch.Tensor, 
                         target: torch.Tensor, 
                         mask: torch.Tensor, 
                         reduction: str ='mean',
                         eps: float = 1e-8):
    """
    Parameters
    ---
    pred: torch.Tensor
        Predicted scores (logits) of shape (B, G, N)
    target: torch.Tensor
        Ground truth of shape (B, G, N). Every (B, N) tensor is a GT binary segmentation mask corresponding to the its category (alphabetic order as the ground truth answer)
    mask: torch.Tensor, optional
        Boolean or float mask of shape (B, G, N), 1=valid, 0=ignore.
    reduction: str
        Reduction method for the loss values. "none", "sum" and "mean" allowed.

    Returns
    ---
    loss: torch.Tensor
        Cross Entropy loss. If `reduction='none'`, loss will be in shape of (B, G) and if `reduction='batch_mean' (B,). If `reduction='mean'` or `'sum'` it will be a scalar.
    """
    # Ensure mask is float
    mask = mask.float()

    # Case 2: target as one-hot
    if target.dim() == 3:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    else:
        raise ValueError("target must have shape (B, G, N)")

    # Apply mask
    loss = loss * mask

    if reduction == 'mean':
        return loss.sum() / mask.sum().clamp_min(1.0)
    elif reduction == 'batch_mean':
        return loss.sum(dim=(1,2)) / (mask.sum(dim=(1,2)) + eps)
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    

def DiceLoss(pred: torch.Tensor,
             target: torch.Tensor, 
             mask: torch.Tensor = None,
             reduction: str = "mean",
             epsilon: float = 1e-6):
    """
    Computes Dice Loss for point cloud segmentation.

    Parameters
    ---
    pred: torch.Tensor
        Predicted scores (logits) of shape (B, G, N)
    target: torch.Tensor
        Ground truth of shape (B, G, N). Every (B, N) tensor is a GT binary segmentation mask corresponding to the its category (alphabetic order as the ground truth answer)
    mask: torch.Tensor, optional
        Boolean or float mask of shape (B, G, N), 1=valid, 0=ignore.
    reduction: str
        Reduction method for the loss values. "none", "sum" and "mean" allowed.
    epsilon: float
        Small value to avoid division by zero.

    Returns
    ---
    loss: torch.Tensor
        Scalar Dice loss (or per-batch loss if reduction='none').
    """

    assert reduction in ["none", "sum", "mean"], "Only 'none', 'mean', 'sum' allowed"
    B, G, N = pred.shape

    mask = mask.float()

    target_one_hot = target.float()

    pred_soft = F.softmax(pred, dim=-1)  # (B, G, N)

    # Apply mask to both pred and target
    pred_soft = pred_soft * mask
    target_one_hot = target_one_hot * mask

    # Dice computation
    intersection = torch.sum(pred_soft * target_one_hot, dim=(1, 2))  # (B,)
    union = torch.sum(pred_soft + target_one_hot, dim=(1, 2))  # (B,)

    dice = (2 * intersection + epsilon) / (union + epsilon)  # (B,)
    loss = 1 - dice  # (B,)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()