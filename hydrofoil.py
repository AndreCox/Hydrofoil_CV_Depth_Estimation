import blenderproc as bproc
import numpy as np
import bpy
import math
import os
import h5py
from tqdm import tqdm
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    """Redirects stdout to devnull to silence Blender's internal logging."""
    fd = sys.stdout.fileno()
    def _redirect_stdout(to_):
        sys.stdout.close()
        os.dup2(to_.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to_=file)
        try:
            yield
        finally:
            _redirect_stdout(to_=old_stdout)

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
bpy.context.scene.cycles.samples = 64

bpy.context.scene.render.use_motion_blur = True
bpy.context.scene.render.motion_blur_shutter = 1 # longer shutter means more blur, shorter means less

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
end_z   = -3.1
num_ride_heights    = 50
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
TRAVEL_DISTANCE = 100.0          # units in X

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def clear_location_keyframes(obj):
    """Remove ALL location fcurves (X, Y, and Z) from the object's action."""
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        to_remove = [fc for fc in action.fcurves if fc.data_path == "location"]
        for fc in to_remove:
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


def write_frame_hdf5(output_dir, frame_idx, colors, z_normalized, hdri_file, hdri_rotation, velocity):
    out_path = os.path.join(output_dir, f"{frame_idx}.hdf5")
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("colors",      data=colors,               compression="gzip")
        hf.create_dataset("ride_height", data=np.float32(z_normalized))
        hf.create_dataset("hdri_source", data=hdri_file)
        hf.create_dataset("hdri_rotation", data=np.float32(hdri_rotation))
        hf.create_dataset("velocity",    data=np.float32(velocity))
    tqdm.write(f"  -> Saved {out_path}")


# ---------------------------------------------------------------------------
# MAIN RENDER LOOP
# ---------------------------------------------------------------------------

frame_counter = 0
origin_x = cylinder.location.x

# Find canvas objects once upfront
canvas_objects = find_canvas_objects()
tqdm.write(f"Found {len(canvas_objects)} Dynamic Paint canvas object(s): {[o.name for o in canvas_objects]}")

# --- Calculate Total Iterations for tqdm ---
total_renders = len(z_values) * len(hdri_files)

# Wrap the outer loop with tqdm
pbar = tqdm(total=total_renders, desc="Rendering Frames", unit="frame", dynamic_ncols=True)

for z, z_normalized in zip(z_values, z_values_normalized):

    cylinder.location.z = z

    for hdri_file in hdri_files:

        tqdm.write(f"\n[Render {frame_counter:04d}] z_norm={z_normalized:.4f} | hdri={hdri_file}")

        # 0. randomize seed for any stochastic processes (e.g. noise texture in shader)
        random_seed = np.random.randint(0, 1_000_000)
        for mat in bpy.data.materials:
            if mat.node_tree:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_NOISE':
                        node.inputs['Seed'].default_value = random_seed

        # also randomize seed for any Geometry Nodes that use it (e.g. for foam distribution)
        for obj in bpy.data.objects:
            for mod in obj.modifiers:
                if mod.type == 'NODES' and mod.node_group:
                    for node in mod.node_group.nodes:
                        if node.type == 'ShaderNodeValue' and "seed" in node.name.lower():
                            node.outputs[0].default_value = random_seed

        # also randomize seed for ocean modifier if present
        for obj in bpy.data.objects:
            for mod in obj.modifiers:
                if mod.type == 'OCEAN':
                    mod.random_seed = random_seed

        # 1. Reset cylinder to start position
        clear_location_keyframes(cylinder)   
        cylinder.location.z = z
        cylinder.location.x = origin_x
        cylinder.location.y = 0.0

        # 2. Reset Dynamic Paint canvas so previous trail doesn't bleed through
        for canvas_obj in canvas_objects:
            reset_dynamic_paint(canvas_obj)
            tqdm.write(f"  Reset Dynamic Paint on: {canvas_obj.name}")

        # 3. Keyframe x: (origin_x - 50) → origin_x over 50 frames
        set_motion_keyframes(cylinder, TOTAL_FRAMES, TRAVEL_DISTANCE, origin_x)

        # 4. Scrub 0 → TOTAL_FRAMES — Dynamic Paint accumulates full trail
        dg = evaluate_scene(frame=RENDER_FRAME)

        # --- 4.5 FREEZE EVALUATED MESH FOR RENDER ---
        frozen_state = {}
        for canvas in canvas_objects:
            # 1. Get the evaluated object
            eval_canvas = canvas.evaluated_get(dg)
            dg.update()
            # 2. Create a brand NEW original mesh from the evaluated object
            # This safely detaches it from the depsgraph's evaluated state
            static_mesh = bpy.data.meshes.new_from_object(
                eval_canvas, 
                preserve_all_data_layers=True, 
                depsgraph=dg
            )

            # 3. Save the original base mesh to restore later
            frozen_state[canvas] = canvas.data

            # 4. Swap the object's mesh to the frozen simulation
            canvas.data = static_mesh

            # 5. Disable modifiers for render so they don't double-evaluate
            for mod in canvas.modifiers:
                mod.show_render = False

        # 5. Sanity check
        t = cylinder.matrix_world.translation
        tqdm.write(f"  Cylinder pos: x={t.x:.2f} y={t.y:.2f} z={t.z:.2f}  (x should be ~{origin_x:.2f})")

        # 6. Camera matrix after full evaluation
        cam_world_matrix = camera.matrix_world.copy()

        # 7. Confirm Geometry Nodes are live
        evaluated_obj = cylinder.evaluated_get(dg)
        mesh = evaluated_obj.to_mesh()
        tqdm.write(f"  Evaluated verts: {len(mesh.vertices)}")
        evaluated_obj.to_mesh_clear()

        # 8. Pin Cycles to RENDER_FRAME
        bpy.context.scene.frame_start   = RENDER_FRAME
        bpy.context.scene.frame_end     = RENDER_FRAME
        bpy.context.scene.frame_current = RENDER_FRAME

        # 9. Set HDRI
        hdri_path = os.path.join(hdri_folder, hdri_file)
        with stdout_redirected():
            bproc.world.set_world_background_hdr_img(hdri_path)


        # Randomize HDRI Z rotation (0 to 360 degrees)
        random_hdri_rotation = np.random.uniform(0, 2 * math.pi)
        world = bpy.context.scene.world
        if world and world.node_tree:
            for node in world.node_tree.nodes:
                if node.type == 'MAPPING':
                    node.inputs['Rotation'].default_value[2] = random_hdri_rotation
                    tqdm.write(f"  HDRI rotation: {math.degrees(random_hdri_rotation):.1f}°")
                    break

        # 10. Register camera pose
        bproc.camera.add_camera_pose(cam_world_matrix)

        # 11. Render
        with stdout_redirected():
            data = bproc.renderer.render()

        # --- 11.5 RESTORE ORIGINAL MESH AND MODIFIERS ---
        for canvas, orig_mesh in frozen_state.items():
            temp_mesh = canvas.data
            canvas.data = orig_mesh
            
            for mod in canvas.modifiers:
                mod.show_render = True
                
            # Free memory
            bpy.data.meshes.remove(temp_mesh)

        # 12. Write HDF5
        with stdout_redirected():
            colors = data["colors"][0]
        write_frame_hdf5(
            output_dir   = output_dir,
            frame_idx    = frame_counter,
            colors       = colors,
            z_normalized = z_normalized,
            hdri_file    = hdri_file,
            hdri_rotation = random_hdri_rotation,
            velocity     = TRAVEL_DISTANCE / TOTAL_FRAMES,
        )

        frame_counter += 1
        pbar.update(1)
        pbar.set_postfix({
            "z_norm": f"{z_normalized:.4f}",
            "hdri": hdri_file,
        })
pbar.close()