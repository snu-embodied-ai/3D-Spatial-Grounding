from isaacsim import SimulationApp

# === Start simulation app ===
simulation_app = SimulationApp({"headless": False})

from asset import AssetCatalog
from sampler import PositionSampler
from camera import CameraManager, CAMERA_POSITIONS
from pathlib import Path

from isaacsim.storage.native import get_assets_root_path


def check(name, test_func, *args):
    try:
        test_func(*args)
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}")
        print(f"      ➜ {e}")

def test_asset():
    isaac_root = Path(get_assets_root_path())
    local_root = Path("/home/choij/isaac-sim/pointcloud/assets")
    asset_catalog = AssetCatalog(isaac_root, local_root)

    print("=== Running AssetCatalog Tests ===")
    check("Isaac asset path", test_asset_root_path, asset_catalog)
    check("Local asset path", test_asset_local_path, asset_catalog)
    check("Spawn height", test_asset_spawn_height, asset_catalog)
    check("Random names", test_asset_random_names_len, asset_catalog)
    print("=== All tests passed ===")

def test_asset_root_path(catalog: AssetCatalog):
    actual = str(catalog.get_path("chef_can"))
    expected = str(Path(get_assets_root_path()) / "Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd")
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_asset_local_path(catalog: AssetCatalog):
    actual = str(catalog.get_path("screws"))
    expected = "/home/choij/isaac-sim/pointcloud/assets/screws/main.usdc"
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_asset_spawn_height(catalog: AssetCatalog):
    actual = catalog.get_spawn_height("newton_craddle")
    expected = 0.1
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_asset_random_names_len(catalog: AssetCatalog):
    actual = len(catalog.random_names(4))
    expected = 4
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_sampler():
    sampler = PositionSampler()

    print("=== Running AssetCatalog Tests ===")
    check("Sample 4", test_sampler_sample_4, sampler)
    check("Sample 5", test_sampler_sample_5, sampler)
    print("=== All tests passed ===")

def test_sampler_sample_4(sampler: PositionSampler):
    actual = len(sampler._sample_indices(4))
    expected = 4
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_sampler_sample_5(sampler: PositionSampler):
    actual = len(sampler._sample_indices(5))
    expected = 5
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_camera():
    manager = CameraManager(width=64, height=48)
    print("=== Running Camera Tests ===")
    check("Manager length", test_camera_manager_len, manager)
    manager.setup()
    check("Setup prim not None", test_camera_setup, manager)
    print("=== All camera tests passed ===")

def test_camera_manager_len(manager: CameraManager):
    actual = len(manager.cameras)
    expected = len(CAMERA_POSITIONS)
    assert actual == expected, f"Expected: {expected}, but got: {actual}"

def test_camera_setup(manager: CameraManager):
    for cam in manager.cameras:
        assert cam.prim is not None, f"Camera {cam.index} prim is None"

test_asset()
test_sampler()
test_camera()