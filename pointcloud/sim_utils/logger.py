import os


NAME_FOR_LOG = {
    "chef_can_u": "chef_can",
    "cracker_box_u": "cracker_box",
    "sugar_box_u": "sugar_box",
    "tomato_soup_can_u": "tomato_soup_can",
    "mustard_bottle_u": "mustard_bottle",
    "tuna_can_u": "tuna_can",
    "pudding_box_u": "pudding_box",
    "gelatin_box_u": "gelatin_box",
    "meat_can_u": "meat_can",
    "bleach_u": "bleach",
    "red_bowl_u": "red_bowl",
    "red_mug_u": "red_mug",
    "power_drill_u": "power_drill",
    "watergun_u": "watergun"
}


class SceneLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, "w") as f:
                f.write("scene_index,name,x,y\n")

    def log(self, scene_index, object_infos):
        for info in object_infos:
            name, position = self._parse_info(info)
            line = f"{scene_index},{NAME_FOR_LOG.get(name, name)},{position[0]:.2f},{position[1]:.2f}\n"
            with open(self.file_path, "a") as f:
                f.write(line)
    
    def _parse_info(self, object_info):
        return object_info.get("name", None), object_info.get("position", None)
