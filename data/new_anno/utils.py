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

def closest_color(requested_color):
    min_colors = {}
    for name in webcolors.names(spec="css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_tuple):
    rgb_tuple *= 255
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value, spec='css3')
    except ValueError:
        # If exact match not found, find the closest color
        return closest_color(rgb_tuple)
    

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
        if "wall" in obj.major_category:
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
    min_score = scores.min()
    max_score = scores.max()
    scores = (scores - min_score) / (max_score - min_score)

    final_output[final_mask] = scores

    return final_output