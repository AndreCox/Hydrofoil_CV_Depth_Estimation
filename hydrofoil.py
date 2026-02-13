import blenderproc as bproc
import numpy as np
import bpy
import math
import os
import cv2

scene = "./scene.blend"
bproc.init()
bproc.loader.load_blend(scene)

# Get hdris in folder
hdri_folder = "./hdris"
hdri_files = [f for f in os.listdir(hdri_folder) if f.endswith('.exr')]
if not hdri_files:
    raise FileNotFoundError(f"No .exr files found in {hdri_folder}")
else:
    print(f"Found HDRI files: {hdri_files}")
        
# 1. Sync Intrinsics
bproc.camera.set_intrinsics_from_blender_params(
    lens=17.5,
    image_width=960,
    image_height=720,
    lens_unit="MILLIMETERS"
)

# 2. Reference Objects
cylinder = bpy.data.objects['Cylinder']
camera = bpy.data.objects['Camera']
cylinder.scale = (1, 1, 1)

# 3. Setup Parenting
camera.parent = cylinder
camera.matrix_parent_inverse = cylinder.matrix_world.inverted()
camera.location = (0, -6.35803, 3.96021)
camera.rotation_euler = (math.radians(56.2051), 0, 0)

# 4. Generate Poses and Keyframes
start_z = cylinder.location.z
end_z = -3.88736
num_ride_heights = 2
z_values = np.linspace(start_z, end_z, num_ride_heights)

# Track metadata for HDF5
used_ride_heights = []
used_hdri_names = []

frame_counter = 0

for z in z_values:
    # Set the physical position of the object
    cylinder.location.z = z
    bpy.context.view_layer.update()
    
    # Get the camera matrix once for this height
    cam_world_matrix = camera.matrix_world.copy()

    for hdri_file in hdri_files:
        # 1. Load the specific HDRI for this frame
        hdri_path = os.path.join(hdri_folder, hdri_file)
        bproc.world.set_world_background_hdr_img(hdri_path)
        
        # 2. Register the camera pose for this specific HDRI/Height combo
        bproc.camera.add_camera_pose(cam_world_matrix)
        
        # 3. Store metadata
        used_ride_heights.append(z)
        used_hdri_names.append(hdri_file)
        
        print(f"Frame {frame_counter}: Z={z:.4f}, HDRI={hdri_file}")
        frame_counter += 1

# 5. Render
# Note: BlenderProc will now render (num_ride_heights * num_hdris) frames
data = bproc.renderer.render()

# 6. ADD CUSTOM DATA TO HDF5
data["ride_height"] = [np.array([z]) for z in used_ride_heights]
data["hdri_source"] = [name for name in used_hdri_names]

# 7. Output
output_dir = "./output"
bproc.writer.write_hdf5(output_dir, data)