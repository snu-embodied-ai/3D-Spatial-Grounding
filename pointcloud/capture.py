import omni.replicator.core as rep
import numpy as np
import os

CAMERA_NUM = 0

# camera choi
camera_path = f"/World/Camera_{CAMERA_NUM}"

# Render Product
camera = rep.get.prim_at_path(camera_path)
render_product = rep.create.render_product(camera, (640, 480))

# Depth Annotator
depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
depth_annotator.attach([render_product])

rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])

# capture depth, rgba data and save it
def capture_depth_and_rgb():
    rep.orchestrator.step_async()
    
    # Depth capture
    depth_data = depth_annotator.get_data()
    if depth_data is not None:
        print("depth shape:", depth_data.shape)
        output_dir = "./choi/pointcloud/data/"
        os.makedirs(output_dir, exist_ok=True)
        # depth save into npy file
        np.save(os.path.join(output_dir, f"depth_frame_{CAMERA_NUM}.npy"), depth_data)
        print(f"Depth frame saved to {output_dir}/depth_frame_{CAMERA_NUM}.npy")
    else:
        print("Failed to capture depth data.")
    
    # RGB capture
    rgb_data = rgb_annotator.get_data()
    if rgb_data is not None:
        # RGB into png file
        rgb_image = np.array(rgb_data, dtype=np.uint8)
        print("rgb shape:" ,rgb_image.shape)
        rgb_output_path = os.path.join(output_dir, f"rgb_frame_{CAMERA_NUM}.png")
        # RGB save
        from PIL import Image
        Image.fromarray(rgb_image).save(rgb_output_path)
        print(f"RGB frame saved to {rgb_output_path}")
    else:
        print("Failed to capture RGB data.")

# capture data
capture_depth_and_rgb()



