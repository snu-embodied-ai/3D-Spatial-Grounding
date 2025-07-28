import argparse
from collections import defaultdict
import webcolors
from spellchecker import SpellChecker
import numpy as np

import unicodedata
import re

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default="/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/new_anno/cfg/dataloader.yaml", help="Path to configuration YAML file")
    parser.add_argument('--num_workers', type=int, help='Number of worker threads to use for processing scenes.')

    args = parser.parse_args()

    return args

def normalize_text(text):
    # Normalize to NFKD form (decompose characters into base + diacritics)
    text = unicodedata.normalize('NFKD', text)
    # Remove diacritics and non-ASCII characters
    text = text.encode('ASCII', 'ignore').decode('ASCII')
    return text

def check_spell(word):
    spell = SpellChecker()

    final_str = ""
    word = normalize_text(word)

    for i, word in enumerate(word.split(" ")):
        prefix = " " if i > 0 else ""
        if word in spell:
            final_str += prefix + word
        else:
            corrected = spell.correction(word)
            final_str += prefix + corrected
    
    return final_str

def find_closest_css_color(rgb_array: np.ndarray):
    """
    Find the closest css colors for each point

    Returns
    ---
        names_per_point: list
            CSS3 color strings for every points
    """

    css3_names = webcolors.names(spec='css3')
    css3_rgb = np.array([webcolors.name_to_rgb(name) for name in css3_names])

    rgb_array *= 255
    
    # rgb_array: (N, 3), css3_rgb: (M, 3)
    diffs = rgb_array[:, None, :] - css3_rgb[None, :, :]  # (N, M, 3)
    dists = np.linalg.norm(diffs, axis=2)  # (N, M)
    idxs = np.argmin(dists, axis=1)  # (N,)

    return [css3_names[i] for i in idxs]
    

def find_duplicates(all_objects: list):
    """
    Find duplicate objects on the same surface and divide them

    Args:
        all_objects (list[pcObject]) : List of objects on same surface

    Returns:
        tuple: A two element tuple containing
            - `divison` (defaultdict(list)) : Labels of duplicate objects
            - `walls` (list) : Wall objects
    """

    division = defaultdict(list)
    walls = []

    for obj in all_objects:
        if obj.major_category is not None and "wall" in obj.major_category:
            walls.append(obj)
        elif obj.final_label is not None:
            division[obj.final_label].append(obj)

    return division, walls


def sample_points(points, num_samples=5):
    if len(points) <= num_samples:
        return points
    idx = np.linspace(0, len(points)-1, num_samples, dtype=int)
    return points[idx]


def apply_gaussian_smoothing(mean: float,
                             std: float,
                             valid_dists: np.ndarray[float],
                             final_mask: np.ndarray[bool]):
    """
    Apply gaussian smoothing to the labels
    Sum Normalization applied for matching the scale and changing to a probabilistc distribution
    """
    final_output = np.zeros_like(final_mask, dtype=float)

    scores = np.exp(-((valid_dists - mean) ** 2) / (2 * std ** 2))
    
    # Min-max scaling
    # min_score = scores.min()
    # max_score = scores.max()
    # scores = (scores - min_score) / (max_score - min_score)

    final_output[final_mask] = scores

    return final_output