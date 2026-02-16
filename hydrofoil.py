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

# Specify EEVEE Renderer
bpy.context.scene.render.engine = 'CYCLES'
# set denoising on and lower the samples
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.samples = 64
        
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
num_ride_heights = 15
z_values = np.linspace(start_z, end_z, num_ride_heights)

# Track metadata for HDF5
used_ride_heights = []
used_hdri_names = []

frame_counter = 0

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
# remove the files in output
for f in os.listdir(output_dir):
    file_path = os.path.join(output_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

for z in z_values:
    cylinder.location.z = z
    bpy.context.view_layer.update()
    cam_world_matrix = camera.matrix_world.copy()

    for hdri_file in hdri_files:
        # 1. Clear previous keyframes/poses so we only render ONE frame
        bproc.utility.reset_keyframes()
        
        # 2. Set HDRI
        hdri_path = os.path.join(hdri_folder, hdri_file)
        bproc.world.set_world_background_hdr_img(hdri_path)
        
        # 3. Add the single pose for this specific frame
        bproc.camera.add_camera_pose(cam_world_matrix)
        
        print(f"Rendering Frame {frame_counter}: Z={z:.4f}, HDRI={hdri_file}")
        
        # 4. Render and write
        data = bproc.renderer.render()
        data["ride_height"] = np.array([[z]], dtype=np.float32) 
        data["hdri_source"] = np.array([[hdri_file]], dtype=object)
        
        # Use a unique prefix or frame index so they don't overwrite
        bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
        
        frame_counter += 1

