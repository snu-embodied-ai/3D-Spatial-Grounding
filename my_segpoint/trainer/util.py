import torch
import torch.nn.functional as F
from typing import List, Dict
import re

def split_by_SEG(generated_texts: list,
                 seg_token: str = "[SEG]"):
    
    all_splits = []

    for text in generated_texts:
        seg_splits = text.split(f"{seg_token}")
        seg_splits = [re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', s) for s in seg_splits]
        all_splits.append(seg_splits[:-1])      # should remove the final split (.)

    return all_splits


def extract_seg_tokens(output_str):
    """
    Extract categories associated with [SEG], ignoring prefixes like 'Mask:' or 'Segmentation:'.
    Handles multi-word category names.
    """
    # Split by [SEG]
    segments = output_str.split('[SEG]')
    
    # Extract last word(s) before [SEG] in each segment
    categories = []
    for seg in segments[:-1]:  # all except the last empty split
        # Remove common prefixes and punctuation
        seg = seg.strip()
        seg = re.sub(r'^(Mask:|Segmentation:|Output:)\s*', '', seg, flags=re.IGNORECASE)
        # Remove trailing punctuation
        seg = seg.rstrip(',:;|')
        categories.append(seg.strip())
    
    return categories

def pad_sequences(all_sequences: List[torch.Tensor]):
    """
    Padding the sequences for stacking into batches

    Parameters
    ---
    all_sequences: list[torch.Tensor]
        List of sequences of different length to stack. Each tensor has shape of `(num_groups, num_points)` where `num_groups` differ among the list elements

    Returns
    ---
    padded_seq: torch.Tensor
        Padded sequence (batched)
    mask: torch.Tensor
        Mask for valid indices (non-padded)
    """

    original_lengths = [seg_h.shape[0] for seg_h in all_sequences]
    max_len = max(original_lengths)

    padded = [F.pad(seq, (0, 0, 0, max_len - seq.shape[0])) for seq in all_sequences]
    padded_seq = torch.stack(padded)  # Shape: (B, max_len, N)

    B, S, N = padded_seq.size()

    mask = torch.zeros((B,S), dtype=bool)
    for orig_len in original_lengths:
        mask[:,:orig_len] = True

    return padded_seq, mask