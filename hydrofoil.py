import blenderproc as bproc
import numpy as np
import bpy
import math
import os
import h5py

scene = "./scene.blend"
bproc.init()
bproc.loader.load_blend(scene)

# --- Get HDRIs in folder ---
hdri_folder = "./hdris"
hdri_files = [f for f in os.listdir(hdri_folder) if f.endswith('.exr')]
if not hdri_files:
    raise FileNotFoundError(f"No .exr files found in {hdri_folder}")

# --- 1. RENDER CONFIGURATION ---
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.samples = 128

bpy.context.scene.render.use_motion_blur = True
bpy.context.scene.render.motion_blur_shutter = 0.5

bpy.context.scene.render.use_simplify = False

# --- 2. CAMERA INTRINSICS ---
bproc.camera.set_intrinsics_from_blender_params(
    lens=17.5,
    image_width=960,
    image_height=720,
    lens_unit="MILLIMETERS"
)

# --- 3. REFERENCE OBJECTS ---
cylinder = bpy.data.objects['Cylinder']
camera   = bpy.data.objects['Camera']
cylinder.scale = (1, 1, 1)

# --- 4. CAMERA PARENTING ---
camera.parent = cylinder
camera.matrix_parent_inverse = cylinder.matrix_world.inverted()
camera.location       = (0, -6.35803, 3.96021)
camera.rotation_euler = (math.radians(56.2051), 0, 0)

# --- 5. RIDE HEIGHT POSES ---
start_z = cylinder.location.z
end_z   = -3.88736
num_ride_heights    = 15
z_values            = np.linspace(start_z, end_z, num_ride_heights)
z_values_normalized = (z_values - start_z) / (end_z - start_z)

# --- 6. OUTPUT DIRECTORY ---
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir(output_dir):
    file_path = os.path.join(output_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

# ---------------------------------------------------------------------------
# MOTION SETTINGS
#
# Cylinder travels x = (origin_x - TRAVEL_DISTANCE) → origin_x over TOTAL_FRAMES.
# Camera is parented so it rides along — the water stays still and the foam
# trail is painted left-to-right across the stationary water mesh in -X.
#
# The key insight: the camera looks along -Y, so X is the screen's left-right
# axis. Travelling in +X means the trail stretches left across the frame.
#
# RENDER_FRAME must always equal TOTAL_FRAMES.
# ---------------------------------------------------------------------------
TOTAL_FRAMES    = 50
RENDER_FRAME    = TOTAL_FRAMES  # never change independently
TRAVEL_DISTANCE = 50.0          # units in X

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def clear_location_keyframes(obj):
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        for fc in [fc for fc in action.fcurves if fc.data_path == "location"]:
            action.fcurves.remove(fc)


def set_motion_keyframes(obj, total_frames, travel_distance, origin_x):
    """
    x at frame F = (origin_x - travel_distance) + (travel_distance / total_frames) * F
    frame 0           → x = origin_x - travel_distance
    frame total_frames → x = origin_x
    All LINEAR so velocity is constant throughout.
    """
    clear_location_keyframes(obj)
    velocity = travel_distance / total_frames
    for frame in range(0, total_frames + 1):
        obj.location.x = (origin_x - travel_distance) + velocity * frame
        obj.keyframe_insert(data_path="location", frame=frame)
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            if fc.data_path == "location":
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


def reset_dynamic_paint(canvas_obj):
    """
    Reset Dynamic Paint canvas by toggling the modifier off and on.
    This clears all accumulated paint data so each render starts with
    a clean canvas — without this, paint from the previous iteration
    bleeds into the next render.
    Does NOT call bpy.ops — safe in headless mode.
    """
    for mod in canvas_obj.modifiers:
        if mod.type == 'DYNAMIC_PAINT':
            mod.show_render   = False
            mod.show_viewport = False
    # Force depsgraph to see the disabled state
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    # Re-enable
    for mod in canvas_obj.modifiers:
        if mod.type == 'DYNAMIC_PAINT':
            mod.show_render   = True
            mod.show_viewport = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()


def find_canvas_objects():
    """Return all objects that have a Dynamic Paint canvas modifier."""
    return [
        obj for obj in bpy.data.objects
        if any(m.type == 'DYNAMIC_PAINT' for m in obj.modifiers)
    ]


def evaluate_scene(frame: int):
    """
    Scrub 0 → frame flushing depsgraph each step.
    Dynamic Paint accumulates paint across every frame — never skip.
    Never call bpy.ops.dpaint.bake() — segfaults headlessly.
    """
    scn = bpy.context.scene
    for f in range(0, frame + 1):
        scn.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def write_frame_hdf5(output_dir, frame_idx, colors, z_normalized, hdri_file, velocity):
    out_path = os.path.join(output_dir, f"{frame_idx}.hdf5")
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("colors",      data=colors,               compression="gzip")
        hf.create_dataset("ride_height", data=np.float32(z_normalized))
        hf.create_dataset("hdri_source", data=hdri_file)
        hf.create_dataset("velocity",    data=np.float32(velocity))
    print(f"  → Saved {out_path}")


# ---------------------------------------------------------------------------
# MAIN RENDER LOOP
# ---------------------------------------------------------------------------

frame_counter = 0
origin_x = cylinder.location.x

# Find canvas objects once upfront
canvas_objects = find_canvas_objects()
print(f"Found {len(canvas_objects)} Dynamic Paint canvas object(s): {[o.name for o in canvas_objects]}")

for z, z_normalized in zip(z_values, z_values_normalized):

    cylinder.location.z = z

    for hdri_file in hdri_files:

        print(f"\n[Render {frame_counter:04d}] z_norm={z_normalized:.4f} | hdri={hdri_file}")

        # 1. Reset cylinder to start position
        cylinder.location.x = origin_x
        cylinder.location.y = 0.0

        # 2. Reset Dynamic Paint canvas so previous trail doesn't bleed through
        for canvas_obj in canvas_objects:
            reset_dynamic_paint(canvas_obj)
            print(f"  Reset Dynamic Paint on: {canvas_obj.name}")

        # 3. Keyframe x: (origin_x - 50) → origin_x over 50 frames
        set_motion_keyframes(cylinder, TOTAL_FRAMES, TRAVEL_DISTANCE, origin_x)

        # 4. Scrub 0 → TOTAL_FRAMES — Dynamic Paint accumulates full trail
        dg = evaluate_scene(frame=RENDER_FRAME)

        # 5. Sanity check
        t = cylinder.matrix_world.translation
        print(f"  Cylinder pos: x={t.x:.2f} y={t.y:.2f} z={t.z:.2f}  (x should be ~{origin_x:.2f})")

        # 6. Camera matrix after full evaluation
        cam_world_matrix = camera.matrix_world.copy()

        # 7. Confirm Geometry Nodes are live
        evaluated_obj = cylinder.evaluated_get(dg)
        mesh = evaluated_obj.to_mesh()
        print(f"  Evaluated verts: {len(mesh.vertices)}")
        evaluated_obj.to_mesh_clear()

        # 8. Pin Cycles to RENDER_FRAME
        bpy.context.scene.frame_start   = RENDER_FRAME
        bpy.context.scene.frame_end     = RENDER_FRAME
        bpy.context.scene.frame_current = RENDER_FRAME

        # 9. Set HDRI
        hdri_path = os.path.join(hdri_folder, hdri_file)
        bproc.world.set_world_background_hdr_img(hdri_path)

        # 10. Register camera pose
        bproc.camera.add_camera_pose(cam_world_matrix)

        # 11. Render
        data = bproc.renderer.render()

        # 12. Write HDF5
        colors = data["colors"][0]
        write_frame_hdf5(
            output_dir   = output_dir,
            frame_idx    = frame_counter,
            colors       = colors,
            z_normalized = z_normalized,
            hdri_file    = hdri_file,
            velocity     = TRAVEL_DISTANCE / TOTAL_FRAMES,
        )

        frame_counter += 1