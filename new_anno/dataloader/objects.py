import re
from glob import glob
import numpy as np

from spellchecker import SpellChecker
import webcolors

from utils import get_color_name, check_spell

class pcObject:
    def __init__(self):
        self._label = None
        self._label_id = None
        self._major_category = None
        self._final_label = None
        self._points = None
        self._color_string = None
        self._idx_in_scene = None
        self._obj_type = None
        self._parent = None
        self._children = []

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, new_label: str):
        self._label = new_label
        return self._label
    
    
    @property
    def label_id(self):
        return self._label_id
    
    @label_id.setter
    def label_id(self, new_label_id: int):
        self._label_id = new_label_id
        return self._label_id
    
    
    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, new_points: np.ndarray):
        self._points = new_points
        return self._points
    

    @property
    def color(self):
        return self._color_string
    
    @color.setter
    def color(self, new_rgb: np.ndarray):
        css_colors = webcolors.names('css3')
        count = np.zeros(len(css_colors))
        for rgb in new_rgb:
            color = get_color_name(rgb)
            count[css_colors.index(color)] += 1

        dominant_color_id = count.argmax()

        self._color_string = css_colors[dominant_color_id]

    
    @property
    def idx_in_scene(self):
        return self._idx_in_scene
    
    @idx_in_scene.setter
    def idx_in_scene(self, new_idx_in_scene: np.ndarray[bool]):
        self._idx_in_scene = new_idx_in_scene
        return self._idx_in_scene
    
    
    @property
    def obj_type(self):
        return self._obj_type
    
    @obj_type.setter
    def obj_type(self, new_obj_type: str):
        self._obj_type = new_obj_type
        return self._obj_type
    
    
    @property
    def major_category(self):
        return self._major_category
    
    @major_category.setter
    def major_category(self, new_major_cat: list[str]):
        self._major_category = new_major_cat
        return self._major_category
    
    
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, supporting_parent):
        assert type(supporting_parent) == pcObject, 'Only pcObject class can be a parent'
        
        self._obj_type = supporting_parent
        return self._parent
    

    @property
    def children(self):
        return self._children
    
    def add_child(self, new_child):
        assert type(new_child) == pcObject, 'Only pcObject class can be a child'

        self._children.append(new_child)
        return self._children
    

    
    def check_label_validity(self):
        exclude_cats = ["wall", "void", "unknown", "unlabeled", "person"]
        misc_cats = ["unk", "objects", "other", "remove", "delete"]

        final_label = None

        # 1. Check if the object major category is valid
        if any(cat in exclude_cats for cat in self.major_category):
            return False

        # 2. Check if the raw label includes words that are miscellaneous
        for exclude in misc_cats:
            if exclude in self.label:
                return False

        # 3. Check if the raw label includes non-string elements
        has_non_alpha = bool(re.search(r'[^a-zA-Z ]', self.label))

        if has_non_alpha:
            final_label = self.major_category[1]
        else:
            final_label = self.label

        # 4. SPELL CHECK - fix typos
        # final_label = check_spell(final_label)

        # 5. Add color information in front of the object label
        if self._color_string is not None:
            final_label = f"{self._color_string} {final_label}"

        self._final_label = final_label
        return final_label
    
    @property
    def final_label(self):
        return self._final_label