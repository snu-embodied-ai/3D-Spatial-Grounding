import numpy as np
import open3d as o3d
import os
import random
import pandas as pd

from copy import deepcopy
import inflect
from scipy.spatial.distance import cdist

RELATIONS_DICT = {
    "view-independent" : ["next to", "besides"],                     # TODO: Add "above", "below"
    "view-dependent" : ["left side of", "right side of", "in front of", "behind"],
    "multiple" : ["between", "duplicate", "mixture"]
}

class Surface():
    def __init__(self,
                 width: float,
                 height: float,
                 thickness: float,
                 grid_size: float,
                 texture: str = None,
                 paint_color: bool = True,
                 density: int = 8):
        """
        - `width`: The x coordinate extent of the surface
        - `height`: The y coordinate extent of the surface
        - `thickness`: The z coordinate extent of the surface
        - `grid_size` : The size of one cell in the grid. The surface's $xy$ plane is transformed to grids to compute the occupancy
        - `texture` : The texture of the surface
        - `paint_color`: Whether to color the surface
        """
        self.width = width
        self.height= height
        self.thickness = thickness
        self.texture = texture
        self.paint_color = paint_color
        self.grid_size = grid_size
        self.density = density

        if self.texture not in ["wood"]:
            raise("Not Implemented yet. Possible option : wood, ")
        
        x_num_cells = np.ceil(self.width / self.grid_size).astype(np.int32)
        y_num_cells = np.ceil(self.height / self.grid_size).astype(np.int32)
        self.occupancy_grid = np.zeros((x_num_cells, y_num_cells))

        
    def load_wood_texture(self):
        mat_data = o3d.data.WoodTexture()

        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = "defaultUnLit"
        self.material.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
        self.material.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
        self.material.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)
    
    def create_mesh(self):
        """
        Create a supporting surface to place the objects
        """
        # The left bottom corner of the surface will be placed at (0,0,0). The right top corner will be located at (width, height, thickness)
        mesh = o3d.geometry.TriangleMesh.create_box(self.width, self.height, 
                                                       self.thickness, create_uv_map=True)
        mesh.compute_vertex_normals()
        mesh = mesh.subdivide_midpoint(self.density)

        self.mesh_dict = {"name": "surface",}
        
        if self.paint_color:
            mesh.paint_uniform_color([0.2, 0.05, 0.0])
        
        if self.texture:
            material = self.load_wood_texture()
            self.mesh_dict['material'] = material

        # self.axis_aligned_bbox = mesh.get_axis_aligned_bounding_box()

        self.mesh_dict['geometry'] = mesh
        
        return self.mesh_dict
    
    def get_mesh(self):
        if self.mesh_dict is None:
            raise Exception("Mesh is not created yet. Create the surface's mesh first")
        else:
            return self.mesh_dict
        
    def get_axis_aligned_bbox(self):
        if self.mesh_dict is None:
            raise Exception("Mesh is not created yet. Create the surface's mesh first")
        else:
            return self.mesh_dict["geometry"].get_axis_aligned_bounding_box()
        
    def get_occupancy_grid(self):
        return self.occupancy_grid
    
    def update_occupancy_grid(self, object_to_place, point_to_place):
        obj_extent = object_to_place.get_axis_aligned_bbox().get_extent()
        x_extent = np.ceil(obj_extent[0] / self.grid_size).astype(np.int32)
        y_extent = np.ceil(obj_extent[1] / self.grid_size).astype(np.int32)

        is_placeable = self.is_placeable(obj_extent, point_to_place)

        x_coord, y_coord = point_to_place
        x = int(x_coord / self.grid_size)
        y = int(y_coord / self.grid_size)
        if is_placeable:
            self.occupancy_grid[x:x+x_extent, y:y+y_extent] = 1
            return True
        else:
            print(f"Cannot place {object_to_place.raw_label} on the surface!")
            return False
        
    def revert_occupancy_grid(self, object_to_place, point_to_place):
        obj_extent = object_to_place.get_axis_aligned_bbox().get_extent()
        x_extent = np.ceil(obj_extent[0] / self.grid_size).astype(np.int32)
        y_extent = np.ceil(obj_extent[1] / self.grid_size).astype(np.int32)

        x_coord, y_coord = point_to_place
        x = int(x_coord / self.grid_size)
        y = int(y_coord / self.grid_size)

        self.occupancy_grid[x:x+x_extent, y:y+y_extent] = 0

    def is_placeable(self, obj_extent, point_to_place):
        x_extent = np.ceil(obj_extent[0] / self.grid_size).astype(np.int32)
        y_extent = np.ceil(obj_extent[1] / self.grid_size).astype(np.int32)

        x_coord, y_coord = point_to_place
        x = int(x_coord / self.grid_size)
        y = int(y_coord / self.grid_size)

        occupany_check = self.occupancy_grid[x:x+x_extent, y:y+y_extent].sum()
        return True if occupany_check == 0 else False

    def get_placeable_region(self, obj_extent):
        x_extent = np.ceil(obj_extent[0] / self.grid_size).astype(np.int32)
        y_extent = np.ceil(obj_extent[1] / self.grid_size).astype(np.int32)

        placeable_points = []

        for x in range(self.occupancy_grid.shape[0] - x_extent):
            for y in range(self.occupancy_grid.shape[1] - y_extent):
                x_coord = x * self.grid_size
                y_coord = y * self.grid_size
                is_placeable = self.is_placeable(obj_extent, np.array((x_coord,y_coord)))
                if is_placeable:
                    placeable_points.append(np.array((x_coord,y_coord)))
        
        return placeable_points

    def draw_heatmap(self, label):
        x_idx, y_idx = np.where(label)
        
        for i in range(x_idx.shape[0]):
            min_x = x_idx[i] / self.grid_size
            min_y = y_idx[i] / self.grid_size
            max_x = (x_idx[i] + 1) / self.grid_size
            max_y = (y_idx[i] + 1) / self.grid_size

            mesh = self.mesh_dict["geometry"]
            points = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)

            valid_x = (min_x <= points[:,0]) * (points[:,0] < max_x)
            valid_y = (min_y <= points[:,1]) * (points[:,1] < max_y)
            label_idx = valid_x * valid_y

            colors[label_idx] = 1 - colors[label_idx]

            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            

# ========================================================================    

class GraspNetObject():
    def __init__(self,
                 category: str,
                 raw_label: str,
                 idx: int,
                 is_main: bool):
        self.category = category
        self.raw_label = raw_label
        self.idx = idx
        self.is_main = is_main

        self.is_distractor = not self.is_main

    def create_mesh(self, graspnet_dir: str):
        graspnet_idx = f"{self.idx:03d}"
        ply_path = os.path.join(graspnet_dir, "models", graspnet_idx, "nontextured_simplified.ply")

        obj_mesh = o3d.io.read_triangle_mesh(ply_path)
        # self.axis_aligned_bbox = obj_mesh.get_axis_aligned_bounding_box()

        self.mesh_dict = {'name': f"GraspNet Object {graspnet_idx}",
                    'geometry': obj_mesh}
        
        return self.mesh_dict
    
    def get_mesh(self):
        if self.mesh_dict is None:
            raise Exception("Mesh is not created yet. Create the object's mesh first")
        else:
            return self.mesh_dict
    
    def get_axis_aligned_bbox(self):
        if self.mesh_dict is None:
            raise Exception("Mesh is not created yet. Create the object's mesh first")
        else:
            return self.mesh_dict["geometry"].get_axis_aligned_bounding_box()
        
    def translate(self, translation: np.ndarray):
        self.mesh_dict["geometry"].translate(translation)

    def get_obb_distance(self, target_obj, use_z: bool = False):
        self_obb_points = np.asarray(self.mesh_dict["geometry"].get_oriented_bounding_box().get_box_points())
        target_obb_points = np.asarray(target_obj.get_mesh()["geometry"].get_oriented_bounding_box().get_box_points())

        self_obb_points = np.unique(self_obb_points, axis=0)
        target_obb_points = np.unique(target_obb_points, axis=0)

        if not use_z:
            self_obb_points = self_obb_points[:,:2]
            target_obb_points = target_obb_points[:,:2]

        distances = cdist(self_obb_points, target_obb_points, metric='euclidean')
        
        return distances

# ========================================================================

class Description():
    def __init__(self, 
                 relation: str,
                 relation_type: str,
                 obj_labels: pd.DataFrame,
                 graspnet_dir: str,
                 between_threshold: int = 0.15,
                 besides_threshold: int = 0.1,
                 describe_freespace: bool = True):
        self.relation = relation
        self.rel_type = relation_type
        self.obj_labels = obj_labels
        self.graspnet_dir = graspnet_dir
        self.describe_freespace = describe_freespace
            
        self.use_raw = True if random.random() > 0.5 else False

        relations = RELATIONS_DICT["view-independent"] + RELATIONS_DICT["view-dependent"]
        if self.relation == "duplicate":
            relations += ["between"]
            self.duplicate_relation = random.choice(relations)
            self.mixture_relations = None
        elif self.relation == "mixture":
            self.mixture_relations = random.sample(relations, 2)
            self.duplicate_relation = None
        else:
            self.duplicate_relation = None
            self.mixture_relations = None

        self.between_threshold = between_threshold
        self.besides_threshold = besides_threshold

        self.failed_to_place_distractor = 0

    def _sample_single_object(self):
        sample = self.obj_labels.sample(1)
        category = sample.iat[0,2]
        raw_label = sample.iat[0,1]
        graspnet_idx = sample.iat[0,0]
        sampled_obj = GraspNetObject(category, raw_label, graspnet_idx, is_main=True)

        return sampled_obj

    def _sample_main_objects(self, num_objects: int) -> list:
        main_objects = []

        for i in range(num_objects):
            sampled_obj = self._sample_single_object()
            if i == 0:
                main_objects.append(sampled_obj)
            else:
                j = 0
                while j < len(main_objects):
                    if self.use_raw:
                        main_label, sample_label = main_objects[j].raw_label, sampled_obj.raw_label
                    else:
                        main_label, sample_label = main_objects[j].category, sampled_obj.category

                    if main_label == sample_label:
                        j = 0
                        sampled_obj = self._sample_single_object()
                    else:
                        j += 1
                main_objects.append(sampled_obj)

        return main_objects

    def create_description(self):
        p = inflect.engine()
        
        if not self.relation == "duplicate" and self.rel_type == "multiple":
            num_main = 2
            self.main_objects = self._sample_main_objects(num_main)

            first_category, first_raw_label = self.main_objects[0].category, self.main_objects[0].raw_label
            second_category, second_raw_label = self.main_objects[1].category, self.main_objects[1].raw_label

            if self.relation == "between":
                if self.use_raw:
                    self.description = f"{self.relation} the {first_raw_label} and the {second_raw_label}"
                else:
                    self.description = f"{self.relation} the {first_category} and the {second_category}"
            else:
                # MIXTURE
                if self.use_raw:
                    self.description = f"{self.mixture_relations[0]} the {first_raw_label} and {self.mixture_relations[1]} the {second_raw_label}"
                else:
                    self.description = f"{self.mixture_relations[0]} the {first_category} and {self.mixture_relations[1]} the {second_category}"

            print(self.description)

        else:
            num_main = 1
            self.main_objects = self._sample_main_objects(num_main)

            first_main = self.main_objects[0]

            if self.relation == "duplicate":
                self.main_objects += deepcopy(self.main_objects)
                if self.duplicate_relation == "between":
                    if self.use_raw:
                        self.description = f"{self.duplicate_relation} the {p.plural(first_main.raw_label)}"
                    else:
                        self.description = f"{self.duplicate_relation} the {p.plural(first_main.category)}"
                else:
                    if self.use_raw:
                        self.description = f"{self.duplicate_relation} the {first_main.raw_label}"
                    else:
                        self.description = f"{self.duplicate_relation} the {first_main.category}"
            else:
                if self.use_raw:
                    self.description = f"{self.relation} the {first_main.raw_label}"
                else:
                    self.description = f"{self.relation} the {first_main.category}"

            print(self.description)


    def get_description(self):
        if self.description is None:
            raise Exception("Description is not created yet. Create a description first!")
        else:
            return self.description
        
    def get_main_objects(self):
        if self.description is None:
            raise Exception("Description is not created yet. Create a description first!")
        else:
            return self.main_objects
        
    def create_main_object_meshes(self):
        if self.description is None:
            raise Exception("Description is not created yet. Create a description first!")
        else:
            for i in range(len(self.main_objects)):
                self.main_objects[i].create_mesh(self.graspnet_dir)
            return self.main_objects
        
    def get_distractors(self):
        if self.description is None:
            raise Exception("Description is not created yet. Create a description first!")
        elif self.main_objects is None:
            raise Exception("Main Objects are not created yet. Create main objects first!")
        else:
            return self.distractors
        
    def create_distractors_meshes(self):
        if self.description is None:
            raise Exception("Description is not created yet. Create a description first!")
        elif self.main_objects is None:
            raise Exception("Main Objects are not created yet. Create main objects first!")
        else:
            for i in range(len(self.distractors)):
                self.distractors[i].create_mesh(self.graspnet_dir)
            return self.distractors
        
    def reset(self):
        print("Setting the description...")
        self.create_description()
        self.create_main_object_meshes()
        
    def place_main_objects(self, surface: Surface):
        surface_max_x, surface_max_y, surface_top = surface.get_axis_aligned_bbox().get_max_bound()

        for i, obj in enumerate(self.main_objects):
            cur_extent = obj.get_axis_aligned_bbox().get_extent()
            cur_min_bound = obj.get_axis_aligned_bbox().get_min_bound()
            from_origin_x = -cur_min_bound[0]
            from_origin_y = -cur_min_bound[1]

            # 1. Z-coordinate translation
            delta_z = surface_top - cur_min_bound[2]
            obj.translate((0,0,delta_z))

            # 2. XY-coordinate translation
            placeable_points = surface.get_placeable_region(cur_extent)
            placeable_points = random.sample(placeable_points, k=len(placeable_points))
            
            if len(placeable_points) == 0:
                print(f"Cannot place {obj.raw_label}. Reset the description")
                return False
            
            # resample = True
            # point_to_place = None
            for point_to_place in placeable_points:
                delta_x = from_origin_x + point_to_place[0]
                delta_y = from_origin_y + point_to_place[1]
                obj.translate((delta_x, delta_y, 0))

                placed = surface.update_occupancy_grid(obj, point_to_place)
                relation_still_valid = self.check_validity(distractor=None, surface=surface, check_main=True)

                if relation_still_valid:
                    break
                else:
                    obj.translate((-delta_x, -delta_y, 0))
                    surface.revert_occupancy_grid(obj, point_to_place)

        return True
    
    def place_single_distractor(self, distractor: GraspNetObject, surface: Surface):
        surface_max_x, surface_max_y, surface_top = surface.get_axis_aligned_bbox().get_max_bound()

        distractor_extent = distractor.get_axis_aligned_bbox().get_extent()
        min_bound = distractor.get_axis_aligned_bbox().get_min_bound()
        from_origin_x = -min_bound[0]
        from_origin_y = -min_bound[1]

        # 1. Z-coordinate translation
        delta_z = surface_top - min_bound[2]
        distractor.translate((0,0,delta_z))

        # 2. XY-coordinate translation
        placeable_points = surface.get_placeable_region(distractor_extent)

        if len(placeable_points) == 0:
            print(f"Cannot place the distractor {distractor.raw_label}. Resampling distractor...")
            return
        else:
            placeable_points = random.sample(placeable_points, k=len(placeable_points))
            failed_to_place = False
            for i in range(len(placeable_points)):
            # while i < len(placeable_points):
                point_to_place = placeable_points[i]
                delta_x = from_origin_x + point_to_place[0]
                delta_y = from_origin_y + point_to_place[1]
                distractor.translate((delta_x, delta_y, 0))

                placed = surface.update_occupancy_grid(distractor, point_to_place)

                relation_still_valid = self.check_validity(distractor, surface)

                if not relation_still_valid:
                    # print(f"Cannot place the distractor {distractor.raw_label} at position ({delta_x}, {delta_y}). Resampling a position to place...")
                    distractor.translate((-delta_x, -delta_y, 0))
                    surface.revert_occupancy_grid(distractor, point_to_place)

                    if i == len(placeable_points) - 1:
                        failed_to_place = True
                else:
                    break
            
            if failed_to_place:
                print(f"Cannot place the distractor {distractor.raw_label} while preserving the relation. Resampling distractor...")
                self.failed_to_place_distractor += 1
                return

        return distractor

    
    def place_all_distractors(self, num_objects: int, surface: Surface):
        distractors = []
        num_distractors = num_objects - len(self.main_objects)

        for i in range(num_distractors):
            resample = True
            while resample:
                distractor = self._sample_single_object()
                distractor.create_mesh(self.graspnet_dir)

                no_duplicate = True
                for main in self.main_objects:
                    if main.raw_label == distractor.raw_label:
                        no_duplicate = False

                if no_duplicate:
                    distractor = self.place_single_distractor(distractor, surface)

                    if distractor is not None:
                        distractors.append(distractor)
                        resample = False
                    else:
                        if self.failed_to_place_distractor < 4:
                            return False

        self.distractors = distractors

        return True

    def _check_validity_single(self, surface: Surface, min, max, relation: str):
        is_valid = True
        threshold_in_grid = int(self.besides_threshold / surface.grid_size)
        occupancy_grid = surface.get_occupancy_grid()
        surface_max_x, surface_max_y = occupancy_grid.shape

        if relation in ["next to", "besides"]:
            min_x = np.max((np.floor(min[0] / surface.grid_size).astype(np.int32) - threshold_in_grid, 0))
            min_y = np.max((np.floor(min[1] / surface.grid_size).astype(np.int32) - threshold_in_grid, 0))
            max_x = np.min((np.ceil(max[0] / surface.grid_size).astype(np.int32) + threshold_in_grid, surface_max_x))
            max_y = np.min((np.ceil(max[1] / surface.grid_size).astype(np.int32) + threshold_in_grid, surface_max_y))
            
        elif relation in ["left side of", "right side of", "in front of", "behind"]:
            min_x = np.floor(min[0] / surface.grid_size).astype(np.int32)
            min_y = np.floor(min[1] / surface.grid_size).astype(np.int32)
            max_x = np.ceil(max[0] / surface.grid_size).astype(np.int32)
            max_y = np.ceil(max[1] / surface.grid_size).astype(np.int32)

            if relation == "left side of":
                if min_x <= 0:
                    return False
                else:
                    min_x = np.max((min_x - threshold_in_grid, 0))
            elif relation == "right side of":
                if max_x >= surface_max_x:
                    return False
                else:
                    max_x = np.min((max_x + threshold_in_grid, surface_max_x))
            elif relation == "in front of":
                if min_y <= 0:
                    return False
                else:
                    min_y = np.min((min_y - threshold_in_grid, 0))
            else:
                if max_y >= surface_max_y:
                    return False
                else:
                    max_y = np.min((max_y + threshold_in_grid, surface_max_y))
                
        else:
            raise NotImplementedError
                        
        besides_region = occupancy_grid[min_x:max_x, min_y:max_y]
        if (besides_region == 1).all():
            is_valid *= False
        else:
            is_valid *= True
                
        return is_valid
    
    def _create_heatmap_label_single(self, relation, surface: Surface, surface_max_x, surface_max_y, occupancy_grid_reversed, all_coord):
        threshold_in_grid = int(self.besides_threshold / surface.grid_size)
        
        if relation in ["next to", "besides"]:
            min_x  = np.max((np.floor(all_coord[0,0] / surface.grid_size).astype(np.int32) - threshold_in_grid, 0))
            min_y = np.max((np.floor(all_coord[1,0] / surface.grid_size).astype(np.int32) - threshold_in_grid, 0))
            max_x = np.min((np.ceil(all_coord[0,1] / surface.grid_size).astype(np.int32) + threshold_in_grid, surface_max_x))
            max_y = np.min((np.ceil(all_coord[1,1] / surface.grid_size).astype(np.int32) + threshold_in_grid, surface_max_y))

        elif relation in ["left side of", "right side of", "in front of", "behind"]:
            min_x  = np.floor(all_coord[0,0] / surface.grid_size).astype(np.int32)
            min_y = np.floor(all_coord[1,0] / surface.grid_size).astype(np.int32)
            max_x = np.ceil(all_coord[0,1] / surface.grid_size).astype(np.int32)
            max_y = np.ceil(all_coord[1,1] / surface.grid_size).astype(np.int32)

            if relation == "left side of":
                min_x = np.max((min_x - threshold_in_grid, 0))
            elif relation == "right side of":
                max_x = np.min((max_x + threshold_in_grid, surface_max_x))
            elif relation == "in front of":
                min_y = np.max((min_y - threshold_in_grid, 0))
            else:           # "behind"
                max_y = np.min((max_y + threshold_in_grid, surface_max_y))
            
        else:
            raise NotImplementedError("Not implemented yet. Only between relation is implemented")
        
        self.label[min_x:max_x, min_y:max_y] = 1
        # Exclude regions occupied by distractors and main objects
        self.label[min_x:max_x, min_y:max_y] = self.label[min_x:max_x, min_y:max_y] *occupancy_grid_reversed[min_x:max_x, min_y:max_y]
                
    
    def check_validity(self, distractor: GraspNetObject, surface: Surface, check_main=False):
        main_objects_aabb_min = [main.get_axis_aligned_bbox().get_min_bound() for main in self.main_objects]
        main_objects_aabb_max = [main.get_axis_aligned_bbox().get_max_bound() for main in self.main_objects]

        if self.rel_type.startswith("view"):
            is_valid = self._check_validity_single(surface, main_objects_aabb_min[0], main_objects_aabb_max[0], self.relation)
            
        elif self.rel_type == "multiple":
            is_valid = True

            if self.relation == "between" or self.duplicate_relation == "between":
                if len(self.main_objects) < 2:
                    is_valid *= True
                elif check_main:
                    first_main = self.main_objects[0]
                    second_main = self.main_objects[1]

                    distance = first_main.get_obb_distance(second_main, use_z=False).min()

                    if self.between_threshold < distance < 3 * self.between_threshold:
                        is_valid *= True
                    else:
                        is_valid *= False
                else:
                    for main in self.main_objects:
                        if (distractor.get_obb_distance(main, use_z=False) < 3 * self.between_threshold).all():
                            is_valid *= False
                        else:
                            is_valid *= True
            
            elif self.relation == "duplicate" and not self.duplicate_relation == "between":
                # Validation check twice
                for min, max in zip(main_objects_aabb_min, main_objects_aabb_max):
                    is_valid *= self._check_validity_single(surface, min, max, self.duplicate_relation)
                
            elif self.relation == "mixture":
                # Validation check twice
                for i in range(len(self.mixture_relations)):
                    min = main_objects_aabb_min[i]
                    max = main_objects_aabb_max[i]
                    mix_rel = self.mixture_relations[i]

                    is_valid *= self._check_validity_single(surface, min, max, mix_rel)
                
            else:
                raise Exception("Not implemented yet for other MULTIPLE relations!")
            
        else:
            raise Exception("No other types implemented - Only View-independent, View-dependent, Multiple")
        
        return is_valid

    def create_heatmap_label(self, surface: Surface):
        occupancy_grid = surface.get_occupancy_grid()
        occupancy_grid_reversed = occupancy_grid == 0
        self.label = np.zeros_like(occupancy_grid)

        surface_max_x, surface_max_y = occupancy_grid.shape

        main_objects_aabb_min = [main.get_axis_aligned_bbox().get_min_bound()[:2] for main in self.main_objects]
        main_objects_aabb_max = [main.get_axis_aligned_bbox().get_max_bound()[:2] for main in self.main_objects]

        all_coord = np.stack(main_objects_aabb_min + main_objects_aabb_max, axis=-1)


        if self.rel_type.startswith("view"):
            self._create_heatmap_label_single(self.relation, surface, surface_max_x, surface_max_y, occupancy_grid_reversed, all_coord[:,:2])
            
        elif self.rel_type == "multiple":
            if self.relation == "between" or self.duplicate_relation == "between":
                all_coord.sort(axis=-1)
                mid_coord = all_coord[:, 1:3]
                if mid_coord[0,0] == mid_coord[0,1]:
                    mid_small_x = np.floor(mid_coord[0,0] / surface.grid_size).astype(np.int32)
                    mid_large_x = mid_small_x + 1
                else:
                    mid_small_x = np.floor(mid_coord[0,0] / surface.grid_size).astype(np.int32)
                    mid_large_x = np.ceil(mid_coord[0,1] / surface.grid_size).astype(np.int32)

                if mid_coord[1,0] == mid_coord[1,1]:
                    mid_small_y = np.floor(mid_coord[1,0] / surface.grid_size).astype(np.int32)
                    mid_large_y = mid_small_y + 1
                else:
                    mid_small_y = np.floor(mid_coord[1,0] / surface.grid_size).astype(np.int32)
                    mid_large_y = np.ceil(mid_coord[1,1] / surface.grid_size).astype(np.int32)
                

                self.label[mid_small_x:mid_large_x, mid_small_y:mid_large_y] = 1
                # Exclude regions occupied by distractors
                self.label[mid_small_x:mid_large_x, mid_small_y:mid_large_y] = self.label[mid_small_x:mid_large_x, mid_small_y:mid_large_y] * \
                                                                        occupancy_grid_reversed[mid_small_x:mid_large_x, mid_small_y:mid_large_y]
                
            elif self.relation == "duplicate" and not self.duplicate_relation == "between":
                for i in range(len(self.main_objects)):
                    idx = 2 * (i+1)
                    self._create_heatmap_label_single(self.duplicate_relation, surface, surface_max_x, surface_max_y, occupancy_grid_reversed, all_coord[:,:idx])
            elif self.relation == "mixture":
                for i in range(len(self.main_objects)):
                    idx = 2 * (i+1)
                    self._create_heatmap_label_single(self.mixture_relations[i], surface, surface_max_x, surface_max_y, occupancy_grid_reversed, all_coord[:,:idx])
            else:
                raise Exception("Not implemented yet. Only between, duplicate, mixture")
        else:
            raise Exception("No other types implemented - Only View-independent, View-dependent, Multiple")

        return self.label

    def save_scene(self, surface: Surface, save_dir: str, ply_index: int, df_label: pd.DataFrame):
        save_dir = os.path.join(save_dir, f"tabletop_{ply_index}.ply")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save mesh
        mesh_path = os.path.join(save_dir, f"tabletop_{ply_index}.ply")

        mesh = o3d.geometry.TriangleMesh()
        for obj in self.main_objects + self.distractors + [surface]:
            mesh += obj.get_mesh()["geometry"]
        
        o3d.io.write_triangle_mesh(mesh_path, mesh)

        # Save descriptions
        df_mesh = pd.DataFrame({"description": self.get_description()}, index=[ply_index])
        df_cat = pd.concat([df_label, df_mesh])

        # Save heatmap labels
        label_path = os.path.join(save_dir, f"{ply_index}_label_grid_size_{surface.grid_size}.npy")
        label = self.create_heatmap_label(surface)
        np.save(label_path, label)

        occupancy_path = os.path.join(save_dir, f"{ply_index}_occupancy_grid_grid_size_{surface.grid_size}.npy")
        np.save(occupancy_path, surface.get_occupancy_grid())

        return df_cat
        


# TODO: should modify the distance computing algorithm -> point cloud distance..? Should only use the xy coordinates...