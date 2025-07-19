import random
from dataclasses import dataclass
from typing import Dict


ASSET_NAMES = [
    "chef_can_u",
    "cracker_box_u",
    "sugar_box_u",
    "tomato_soup_can_u",
    "mustard_bottle_u",
    "tuna_can_u",
    "pudding_box_u",
    "gelatin_box_u",
    "meat_can_u",
    "banana",
    "bleach_u",
    "red_bowl_u",
    "red_mug_u",
    "power_drill_u",
    "wood_block",
    "scissors",
    "marker",
    "clamp",
    "brick",
    "green_mug",
    "black_mug",
    "yellow_mug",
    "blue_mug",
    "red_block",
    "green_block",
    "blue_block",
    "yellow_block",
    "nvidia_cube",
    "rubiks_cube",
    "mac_n_cheese",
    # belows are local
    "screws",
    "checkerboard",
    "alarm_clock",
    "apple",
    "avocado",
    "brass_vase",
    "camera",
    "copper_scale",
    "eraser",
    "fork",
    "knife",
    "lemon",
    "lichees",
    "lime",
    "newton_craddle",
    "orange",
    "pencil",
    "pomegranate",
    "pumpkin",
    "red_onion",
    "spoon",
    "sketchbook",
    "strawberries",
    "watergun_u",
    "wooden_bowl",
]


ASSET_NAME_TO_PATH = {
    "chef_can": "/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
    "chef_can_u": "/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
    "cracker_box": "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    "cracker_box_u": "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    "sugar_box": "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    "sugar_box_u": "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    "tomato_soup_can": "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "tomato_soup_can_u": "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
    "mustard_bottle": "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "mustard_bottle_u": "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "tuna_can_u": "/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
    "pudding_box": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    "pudding_box_u": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
    "gelatin_box": "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
    "gelatin_box_u": "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
    "meat_can": "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    "meat_can_u": "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
    "banana": "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
    "bleach": "/Isaac/Props/YCB/Axis_Aligned/021_bleach_cleanser.usd",
    "bleach_u": "/Isaac/Props/YCB/Axis_Aligned/021_bleach_cleanser.usd",
    "red_bowl_u": "/Isaac/Props/YCB/Axis_Aligned/024_bowl.usd",
    "red_mug_u": "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
    "power_drill": "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd",
    "power_drill_u": "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd",
    "wood_block": "/Isaac/Props/YCB/Axis_Aligned/036_wood_block.usd",
    "scissors": "/Isaac/Props/YCB/Axis_Aligned/037_scissors.usd",
    "marker": "/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd",
    "clamp": "/Isaac/Props/YCB/Axis_Aligned/051_large_clamp.usd",
    "brick": "/Isaac/Props/YCB/Axis_Aligned/061_foam_brick.usd",
    "green_mug": "/Isaac/Props/Mugs/SM_Mug_A2.usd",
    "black_mug": "/Isaac/Props/Mugs/SM_Mug_B1.usd",
    "yellow_mug": "/Isaac/Props/Mugs/SM_Mug_C1.usd",
    "blue_mug": "/Isaac/Props/Mugs/SM_Mug_D1.usd",
    "red_block": "/Isaac/Props/Blocks/red_block.usd",
    "green_block": "/Isaac/Props/Blocks/green_block.usd",
    "blue_block": "/Isaac/Props/Blocks/blue_block.usd",
    "yellow_block": "/Isaac/Props/Blocks/yellow_block.usd",
    "nvidia_cube": "/Isaac/Props/Blocks/nvidia_cube.usd",
    "rubiks_cube": "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd",
    "mac_n_cheese": "/Isaac/Props/Food/mac_n_cheese_centered.usd",
    # belows are local
    "screws": "/screws/main.usdc",
    "checkerboard": "/checkerboard/main.usdc",
    "alarm_clock": "/alarm_clock/main.usdc",
    "apple": "/apple/main.usdc",
    "avocado": "/avocado/main.usdc",
    "brass_vase": "/brass_vase/main.usdc",
    "camera": "/camera/main.usdc",
    "copper_scale": "/copper_scale/main.usdc",
    "eraser": "/eraser/main.usdc",
    "fork": "/fork/main.usdc",
    "knife": "/knife/main.usdc",
    "lemon": "/lemon/main.usdc",
    "lichees": "/lichees/main.usdc",
    "lime": "/lime/main.usdc",
    "newton_craddle": "/newton_craddle/main.usdc",
    "orange": "/orange/main.usdc",
    "pencil": "/pencil/main.usdc",
    "pomegranate": "/pomegranate/main.usdc",
    "pumpkin": "/pumpkin/main.usdc",
    "red_onion": "/red_onion/main.usdc",
    "spoon": "/spoon/main.usdc",
    "sketchbook": "/sketchbook/main.usdc",
    "strawberries": "/strawberries/main.usdc",
    "watergun_u": "/watergun/main.usdc",
    "wooden_bowl": "/wooden_bowl/main.usdc",
}

ASSET_NAME_TO_SPAWN_HEIGHT = {
    "chef_can": 0.15,
    "chef_can_u": 0.17,
    "cracker_box": 0.135,
    "cracker_box_u": 0.205,
    "sugar_box": 0.125,
    "sugar_box_u": 0.19,
    "tomato_soup_can": 0.135,
    "tomato_soup_can_u": 0.15,
    "mustard_bottle": 0.13,
    "mustard_bottle_u": 0.195,
    "tuna_can_u": 0.12,
    "pudding_box": 0.115,
    "pudding_box_u": 0.14,
    "gelatin_box": 0.115,
    "gelatin_box_u": 0.14,
    "meat_can": 0.13,
    "meat_can_u": 0.14,
    "banana": 0.13,
    "bleach": 0.13,
    "bleach_u": 0.225,
    "red_bowl_u": 0.13,
    "red_mug_u": 0.14,
    "power_drill": 0.125,
    "power_drill_u": 0.19,
    "wood_block": 0.15,
    "scissors": 0.11,
    "marker": 0.11,
    "clamp": 0.11,
    "brick": 0.125,
    "green_mug": 0.1,
    "black_mug": 0.1,
    "yellow_mug": 0.1,
    "blue_mug": 0.1,
    "red_block": 0.125,
    "green_block": 0.125,
    "blue_block": 0.125,
    "yellow_block": 0.125,
    "nvidia_cube": 0.14,
    "rubiks_cube": 0.14,
    "mac_n_cheese": 0.12,
    # belows are local
    "screws": 0.105,
    "checkerboard": 0.105,
    "alarm_clock": 0.1,
    "apple": 0.1,
    "avocado": 0.1,
    "brass_vase": 0.1,
    "camera": 0.1,
    "copper_scale": 0.1,
    "eraser": 0.11,
    "fork": 0.1,
    "knife": 0.1,
    "lemon": 0.15,
    "lichees": 0.1,
    "lime": 0.1,
    "newton_craddle": 0.1,
    "orange": 0.16,
    "pencil": 0.105,
    "pomegranate": 0.1,
    "pumpkin": 0.1,
    "red_onion": 0.1,
    "spoon": 0.1,
    "sketchbook": 0.1,
    "strawberries": 0.1,
    "watergun_u": 0.11,
    "wooden_bowl": 0.1,
}


LOCAL_ASSET_ROOT_PATH = "/home/choij/isaac-sim/pointcloud/assets"


@dataclass
class Asset:
    name: str
    usd_path: str
    spawn_height: float


class AssetCatalog:
    def __init__(self, isaac_root: str, local_root: str):
        self.assets: Dict[str, Asset] = {}
        self.isaac_root = isaac_root
        self.local_root = local_root

        for name in ASSET_NAMES:
            relative_path = ASSET_NAME_TO_PATH[name]
            spawn_height = ASSET_NAME_TO_SPAWN_HEIGHT[name]

            full_path = self._get_full_path(relative_path)

            self.assets[name] = Asset(
                name=name,
                usd_path=full_path,
                spawn_height=spawn_height
            )
    
    def _get_full_path(self, relative_path):
        if relative_path.startswith("/Isaac"):
            root_path = self.isaac_root
        else:
            root_path = self.local_root
        return root_path + relative_path
    
    def get_path(self, name):
        return self.assets[name].usd_path

    def get_spawn_height(self, name):
        return self.assets[name].spawn_height

    def random_names(self, count):
        return random.sample(list(self.assets.keys()), count)
