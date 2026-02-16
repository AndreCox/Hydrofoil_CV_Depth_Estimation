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

# --- 1. MOTION BLUR CONFIGURATION ---
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.samples = 64

# Enable Motion Blur
bpy.context.scene.render.use_motion_blur = True
# 0.5 is standard shutter. Increase toward 1.0 for more "streak"
bpy.context.scene.render.motion_blur_shutter = 0.5 

# 2. Sync Intrinsics
bproc.camera.set_intrinsics_from_blender_params(
    lens=17.5,
    image_width=960,
    image_height=720,
    lens_unit="MILLIMETERS"
)

# 3. Reference Objects
cylinder = bpy.data.objects['Cylinder']
camera = bpy.data.objects['Camera']
cylinder.scale = (1, 1, 1)

# 4. Setup Parenting
camera.parent = cylinder
camera.matrix_parent_inverse = cylinder.matrix_world.inverted()
camera.location = (0, -6.35803, 3.96021)
camera.rotation_euler = (math.radians(56.2051), 0, 0)

# 5. Generate Poses
start_z = cylinder.location.z
end_z = -3.88736
num_ride_heights = 15
z_values = np.linspace(start_z, end_z, num_ride_heights)
z_values_normalied = (z_values - start_z) / (end_z - start_z)

frame_counter = 0
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Clean output dir
for f in os.listdir(output_dir):
    file_path = os.path.join(output_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

# --- 6. RENDER LOOP ---
# Forward velocity: How much the boat moves per frame (in meters)
# Adjust this to change the intensity of the blur
base_velocity = 1

for z, z_normalized in zip(z_values, z_values_normalied):
    # Set current ride height
    cylinder.location.z = z
    
    # We must update the view layer to get the correct matrix for the camera
    bpy.context.view_layer.update()
    cam_world_matrix = camera.matrix_world.copy()

    for hdri_file in hdri_files:
        # Clear previous frame data
        bproc.utility.reset_keyframes()
        
        # MOTION BLUR LOGIC:
        # We set two keyframes for the cylinder. 
        # Frame -1: Current Position - Velocity
        # Frame 1: Current Position + Velocity
        cylinder.location.x = -base_velocity
        cylinder.keyframe_insert(data_path="location", frame=-1)
        
        cylinder.location.x = base_velocity
        cylinder.keyframe_insert(data_path="location", frame=1)

        bpy.context.view_layer.update()
        cam_world_matrix = camera.matrix_world.copy()
        
        # Set HDRI
        hdri_path = os.path.join(hdri_folder, hdri_file)
        bproc.world.set_world_background_hdr_img(hdri_path)
        
        # Set camera pose (it moves with the cylinder because it's parented)
        bproc.camera.add_camera_pose(cam_world_matrix)
        
        print(f"Rendering Frame {frame_counter}: Z_norm={z_normalized:.4f}, Blur_Vel={base_velocity}")
        
        # Render
        data = bproc.renderer.render()
        
        # Write Metadata
        data["ride_height"] = np.array([[z_normalized]], dtype=np.float32) 
        data["hdri_source"] = np.array([[hdri_file]], dtype=object)
        data["velocity"] = np.array([[base_velocity]], dtype=np.float32)
        
        bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
        
        frame_counter += 1