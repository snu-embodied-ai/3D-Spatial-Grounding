import torch

def compute_mIoU_binary_groups(preds, targets, valid_mask, threshold=0.5):
    """
    Parameters
    ---
    preds: torch.Tensor
        (B, G, N) predicted scores (probabilities or logits)
    targets: torch.Tensor
        (B, G, N) binary ground truth masks
    valid_mask: torch.Tensor
        (B, G) boolean or 0/1 mask for valid groups
    threshold: float
        threshold for binarizing preds if not already binary

    Returns
    ---
        mIoU: torch.Tensor
            per batch scalar mean IoU over all valid groups
        iou_per_group: torch.Tensor
            (B, G) IoU values (NaN for invalid groups)
    """
    # If preds are probabilities/logits, threshold them
    preds_bin = (preds >= threshold).to(torch.bool)
    targets_bin = (targets >= 0.5).to(torch.bool)

    # Intersection and union over N points
    intersection = (preds_bin & targets_bin).sum(dim=-1).float()  # (B, G)
    union = (preds_bin | targets_bin).sum(dim=-1).float()         # (B, G)

    # IoU per (B, G)
    iou = intersection / union
    iou[union == 0] = float('nan')  # avoid div-by-zero

    # Mask invalid groups
    valid_mask = valid_mask.to(torch.bool)
    iou[~valid_mask] = float('nan')

    # Mean over batches
    mIoU = torch.nanmean(iou, dim=1)

    return mIoU, iou


def compute_mIoU_vectorized(preds, targets, num_classes):
    """
    preds: (B, N) predicted class indices
    targets: (B, N) ground truth class indices
    num_classes: int, total number of classes
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Filter out invalid labels (optional: e.g., if -1 is "ignore")
    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    # Compute confusion matrix
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=preds.device)
    indices = num_classes * targets + preds
    conf_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)

    # IoU = TP / (TP + FP + FN)
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(0) - tp
    fn = conf_matrix.sum(1) - tp
    denom = tp + fp + fn

    iou = tp.float() / denom.float()
    iou[denom == 0] = float('nan')  # ignore classes with no samples

    return torch.nanmean(iou).item(), iou  # returns mIoU and per-class IoU