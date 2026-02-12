import blenderproc as bproc
import numpy as np
import bpy
import math
import os
import cv2

scene = "./scene.blend"
bproc.init()
bproc.loader.load_blend(scene)

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
num_samples = 2
z_values = np.linspace(start_z, end_z, num_samples)

if cylinder.animation_data:
    cylinder.animation_data_clear()

for i, z in enumerate(z_values):
    cylinder.location.z = z
    cylinder.keyframe_insert(data_path="location", index=2, frame=i)
    bpy.context.view_layer.update()
    
    cam_world_matrix = camera.matrix_world.copy()
    bproc.camera.add_camera_pose(cam_world_matrix)
    print(f"Frame {i}: Ride Height (Z) = {z:.4f}")

# 5. Render
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
data = bproc.renderer.render()

# 6. ADD CUSTOM DATA TO HDF5
# We add the ride height array to the data dictionary. 
# bproc.writer.write_hdf5 will automatically save this as a dataset in the .h5 file.
data["ride_height"] = [np.array([z]) for z in z_values]

# 7. Output
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Now this will work without the Exception
bproc.writer.write_hdf5(output_dir, data)

# Visual check: Save PNGs
for i, image in enumerate(data['colors']):
    img_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"frame_{i}.png"), img_bgr)

print(f"Success! 'ride_height' has been saved into the HDF5 file in {output_dir}")