from collections import defaultdict
from typing import Dict, List

import numpy as np

from objects import IsaacSimObject

# def refine_object_groups(all_objects: List[List[Dict]]) -> List[Dict]:
#     """
    
#     """
#     final_objects = []
#     cam0_objs, cam1_objs, cam2_objs = all_objects

#     cam0_supporters = [group["supporting"] for group in cam0_objs]
#     cam1_supporters = [group["supporting"] for group in cam1_objs]
#     cam2_supporters = [group["supporting"] for group in cam2_objs]
#     supporters = set(cam0_supporters + cam1_supporters + cam2_supporters)

#     cam0_objects_on = [group["objects_on"] for group in cam0_objs]
#     cam1_objects_on = [group["objects_on"] for group in cam1_objs]
#     cam2_objects_on = [group["objects_on"] for group in cam2_objs]

#     for supp in supporters:
#         objects_on = set()
#         # 1. Get from cam0
#         if supp in cam0_supporters:
#             idx0 = cam0_supporters.index(supp)
#             objects_on.add(cam0_objects_on[idx0])
        
#         # 2. Get from cam1
#         if supp in cam1_supporters:
#             idx1 = cam1_supporters.index(supp)
#             objects_on.add(cam1_objects_on[idx1])

#         # 3. Get from cam2
#         if supp in cam2_supporters:
#             idx2 = cam2_supporters.index(supp)
#             objects_on.add(cam2_objects_on[idx2])

#         final_objects.append({
#             "supporting": supp,
#             "objects_on": list(objects_on)
#         })

#     return final_objects


def load_single_object(obj_name: str,
                       results: dict):
    # 1. Find object id
    obj_id = int(results["label_to_id"][obj_name])
    # if "table" in obj_name:
    #     obj_id = results["semantic_LabelToID"]["UNLABELLED"]
    # elif obj_name in results["semantic_LabelToID"].keys():
    #     obj_id = results["semantic_LabelToID"][obj_name]
    # elif f"{obj_name}_u" in results["semantic_LabelToID"].keys():
    #     obj_id = results["semantic_LabelToID"][f"{obj_name}_u"]
    # else:
    #     raise ValueError(f"{obj_name} not in semantic labels!")
    # obj_id = int(obj_id)
    
    print(f"OBJ ID : {obj_id}")
    semantic_mask = results["segments"][:,0] == obj_id
    instances_mask = results["segments"][:,1][semantic_mask]

    unique_instance_ids, counts = np.unique(instances_mask, return_counts=True)
    print(unique_instance_ids, counts)
    all_instances = []
    for i, inst_id in enumerate(unique_instance_ids):
        inst_mask = results["segments"][:,1] == inst_id
        inst_feats = results["pcd_feats"][inst_mask]

        instance = IsaacSimObject(obj_name, inst_feats, inst_mask)
        all_instances.append(instance)

    return all_instances

# def load_all_objects(all_objects: List[List[Dict]],
#                      results: dict) -> Dict[list]:
#     """
#     FIXME:
#     fix this to create structure as a I intended to
    
#     """
#     objects_in_scene = defaultdict(list)
#     supporters = defaultdict(list)

#     # 1. Get three `all_objects` and refine them into a single `all_objects`
#     assert len(all_objects) == 3, Exception("Only 3 views were intended to be accepted!!")
#     all_objects = refine_object_groups(all_objects)

#     for group in all_objects:
#         support = group["support"]
#         objects_on = group["objects_on"]

#         # 2. Generate support objects
#         if support in objects_in_scene:
#             support_insts = objects_in_scene[support]
#         else:
#             support_insts = load_single_object(support, True, results)
        
#         objects_on_insts = defaultdict(list)
#         for obj_name in objects_on:
#             if obj_name in objects_in_scene:
#                 objects_on_insts[obj_name] = objects_in_scene[obj_name]
#             else:
#                 objects_on_insts[obj_name] = load_single_object(obj, False, results)

#         # 3. Refine support-supported relation (match instances)
#         # Select one support instance
#         for supp in support_insts:
#             # Add support instance to dictionary
#             supporters[supp.name].append(supp)
#             objects_in_scene[supp.name].append(supp)

#             # Select one instance among the objects (supported by)
#             for obj_name, obj_instances in objects_on_insts.itmes():
#                 for instance in obj_instances:
#                     if supp.set_objects_on_surface(instance):
#                         pass
                    

#     for obj in all_objects:
#         if "table" in obj:
#             obj_id = results["semantic_LabelToID"]["UNLABELLED"]
#         elif obj in results["semantic_LabelToID"].keys():
#             obj_id = results["semantic_LabelToID"][obj]
#         elif f"{obj}_u" in results:
#             obj_id = results["semantic_LabelToID"][f"{obj}_u"]
#         else:
#             raise ValueError(f"{obj} not in semantic labels!")
        
#         semantic_mask = results["segments"][:,0] == obj_id
#         instances_mask = results["segments"][:,1][semantic_mask]

#         unique_instance_ids = np.unique(instances_mask)
#         all_instances = []
#         for i, inst_id in enumerate(unique_instance_ids):
#             inst_mask = results["segments"][:,1] == inst_id
#             inst_feats = results["pcd_feats"][inst_mask]
#             if len(unique_instance_ids) > 1:
#                 obj_name = f"{obj}_{i}"
#             else:
#                 obj_name = obj

#             instance = IsaacSimObject(obj_name, inst_feats, inst_mask)
#             all_instances.append(instance)

#         object_data[obj] = all_instances

#     return object_data



# REVISED VERSION
# Computing support/supported-by relations by distance metrics, not relying on GPT


# FIXME:
# CHANGE THIS ALGORITHM to ray shooting. Check if collision occur and determine if it is a supporting object

def find_supported(support: List[IsaacSimObject],
                   instances: List[IsaacSimObject]):
    """
    FIXME:
    facing local minima
    should switch to another branch,
    But HOW?
    Sol 1. Shoot UPWARD rays from an object and check if any collision occur (this could be cheaper since only computation runs n times)
    Sol 2. Run this algorithm for all pairs. In other words, increase the operation from O(nlogn) to O(n^2)
    """
    supported_by = []
    remaining_objs = instances

    for sup in support:
        sup.detect_support_surface()
        # TODO: Add code if any object is placed on this supporter object.
        # OR, check if `supported_by` is empty. If empty,
        # No, this cannot
        
        leftover = []
        for inst in remaining_objs:
            if sup.set_objects_on_surface(inst):
                supported_by.append(inst)
            else:
                leftover.append(inst)

        remaining_objs = leftover
    
    # if len(supported_by) == 0:
    #     pass

    return supported_by, remaining_objs

def generate_tree(bottom_instance: IsaacSimObject,
                  instances: List[IsaacSimObject]):
    """
    Generate full tree of support / supported-by relationship
    """
    support_objs = [bottom_instance]
    remaining = instances

    while len(remaining) > 0:
        print(f"supp: {[obj.name for obj in support_objs]}")
        print(f"rem: {[obj.name for obj in remaining]}")
        support_objs, remaining = find_supported(support_objs, remaining)


def load_and_group_objects(data: dict):

    """
    Rebuilt object loader function
    """

    # 1. Load names of all object instances in the scene
    object_names = data["label_to_id"].keys()
    print(object_names)
    instances = []
    objects_dict = defaultdict(list)
    for name in object_names:
        print(name)
        obj_insts = load_single_object(name, data)
        instances += obj_insts
        objects_dict[name] = obj_insts

    print("Finished loading all object points")

    # 2. Check if instances are loaded correctly
    for name, insts_in_obj in objects_dict.items():
        assert len(insts_in_obj) == (np.array(list(object_names)) == name).sum(), AttributeError(f"The number of instances of {name} generated from segmentation masks are different from the .log file !!")

    # 3. Find the lowest object (bottom object) and pop this out from the instances list
    #    -> which will be the table object in IsaacSim Tabletop Dataset
    z_min_of_insts = np.array([inst.inst_feats[:,2].min() for inst in instances])
    lowest_idx = z_min_of_insts.argmin()
    bottom_instance = instances[lowest_idx]
    instances.pop(lowest_idx)
    
    # 4. Set this instance as the initial supporter (root node) and find other instances supported by this instance
    generate_tree(bottom_instance, instances)
    instances.append(bottom_instance)

    return objects_dict, instances