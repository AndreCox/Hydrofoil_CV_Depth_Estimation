import h5py
import io
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def create_filtered_gif(hdf5_path, output_gif, bg_name=None, step=5, fps=12):
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 1. Identify valid indices
            heights = f['ride_height'][...]
            # Find the background key (it was the one throwing string errors earlier)
            bg_key = [k for k in f.keys() if 'source' in k.lower() or 'hdri_file' in k.lower()][0]
            bg_data = [val.decode('utf-8') if isinstance(val, bytes) else val for val in f[bg_key][...]]
            
            # If no background is specified, include frames from every background.
            if bg_name is None:
                valid_indices = list(range(len(bg_data)))
                bg_label = "all backgrounds"
                print(f"No background specified. Using {bg_label}.")
            else:
                # Filter indices where background matches
                valid_indices = [i for i, name in enumerate(bg_data) if name == bg_name]
                bg_label = bg_name

            # Sort these specific indices by ride_height
            sorted_indices = np.array(valid_indices)[np.argsort(heights[valid_indices])]
            final_indices = sorted_indices[::step]
            
            if len(final_indices) == 0:
                print(f"Error: No frames found for background '{bg_label}'")
                return

            print(f"Creating GIF for '{bg_label}' with {len(final_indices)} frames...")

            frames = []
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            except:
                font = ImageFont.load_default()

            for idx in tqdm(final_indices, desc="Processing Frames", colour='magenta'):
                raw_bytes = f['colors_webp'][idx].tobytes()
                img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
                
                # Draw Overlay
                draw = ImageDraw.Draw(img)
                h_val = f['ride_height'][idx]
                v_val = f['velocity'][idx]
                frame_bg_name = f[bg_key][idx]
                overlay_text = f"Height: {h_val:.3f}m | Vel: {v_val:.1f}m/s\nBG: {frame_bg_name}"
                
                # Draw a simple shadow/outline for readability
                draw.text((22, 22), overlay_text, fill="black", font=font)
                draw.text((20, 20), overlay_text, fill="white", font=font)
                
                frames.append(img)

            # 2. Save
            frames[0].save(
                output_gif,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0
            )
            print(f"Success! Saved to {output_gif}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtered Hydrofoil Animator")
    parser.add_argument("--file", "-f", default="hydrofoil_webp.hdf5")
    parser.add_argument("--bg", help="Specific background name to filter by. Defaults to all backgrounds.")
    parser.add_argument("--step", "-s", type=int, default=5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", "-o", default="filtered_anim.gif")

    args = parser.parse_args()
    create_filtered_gif(args.file, args.out, args.bg, args.step, args.fps)