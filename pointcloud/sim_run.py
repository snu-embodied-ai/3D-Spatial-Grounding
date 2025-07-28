import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scenes", type=int, default=500)
    parser.add_argument("--rgbd_dir", type=str, default="./pointcloud/rgbd")
    parser.add_argument("--pose_dir", type=str, default="./pointcloud/pose")
    parser.add_argument("--log_path", type=str, default="./pointcloud/log")
    parser.add_argument("--asset_list", type=str, default="",
                        help="Select among full, isaac, local, mug and block")
    parser.add_argument("--sampler", type=str, default="random",
                        help="Select among random and aligned")
    parser.add_argument("--spawner", type=str, default="unique",
                        help="Select among unique and duplicate")
    parser.add_argument("--num_objects", type=int, default=4)
    parser.add_argument("--target_object", type=str, default="banana")
    return parser.parse_args()

def start_simulation(args):
    from isaacsim import SimulationApp

    # Start Simulation App
    simulation_app = SimulationApp({"headless": args.headless})
    return simulation_app

def do_simulation(args, simulation_app):
    import os
    from isaacsim.core.api import World
    from sim_utils.samplers import create as create_sampler
    from sim_utils.spawners import create as create_spawner
    from sim_utils.cameras import CameraManager
    from sim_utils.loggers import SceneLogger
    from sim_utils.scenes import generate_ground_plane, generate_light, generate_table

    # Save Directory
    rgbd_dir = args.rgbd_dir
    pose_dir = args.pose_dir
    log_path = args.log_path

    # Basic Scene
    generate_ground_plane()
    generate_light()
    table = generate_table()

    # Asset, Sampler, Builer, Camera Manager
    catalog = create_asset_catalog(args.asset_list)
    sampler = create_sampler(args.sampler)
    spawner = create_spawner(args.spawner,
                             catalog=catalog,
                             sampler=sampler,
                             num_objects=args.num_objects,
                             target_object=args.target_object)
    cam_manager = CameraManager()
    logger = SceneLogger(log_path)

    # Setup Camera & Save Poses
    cam_manager.setup()
    os.makedirs(pose_dir, exist_ok=True)
    cam_manager.save_all_poses(pose_dir)

    # Main Simulation Loop
    os.makedirs(rgbd_dir, exist_ok=True)
    world = World()
    world.reset()

    scene_index = 0
    spawn_interval = 2
    num_frames = args.scenes * 2

    print("[Main] Simulation start")
    for frame in range(num_frames):
        if frame % spawn_interval == 0:
            spawner.clear_objects()
            table.set_random_color()

            world.step(render=True)
            spawner.spawn_random()
            object_info = spawner.get_spawned_object_info()
            logger.log(scene_index, object_info)
            print(f"[Main] Spawned {len(object_info)} objects at scene {scene_index}")

            for _ in range(15):
                world.step(render=True)
            cam_manager.capture_all(rgbd_dir, scene_index)
            scene_index += 1

        world.step(render=True)

    print("[Main] Simulation terminated")
    simulation_app.close()

def create_asset_catalog(asset_list):
    from isaacsim.storage.native import get_assets_root_path
    from sim_utils.assets.asset import AssetCatalog
    from sim_utils.assets import create as create_asset_list
    asset_list = create_asset_list(asset_list)
    catalog = AssetCatalog(get_assets_root_path(), asset_list=asset_list)
    return catalog

def main():
    args = parse_args()
    simulation_app = start_simulation(args)
    do_simulation(args, simulation_app)


if __name__ == "__main__":
    main()
