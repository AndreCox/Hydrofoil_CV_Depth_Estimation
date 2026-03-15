import h5py
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def inspect_sample(hdf5_path, index, show_image=True):
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'colors_webp' not in f:
                print(f"Error: {hdf5_path} does not contain 'colors_webp'.")
                return

            total_samples = len(f['colors_webp'])
            if index < 0 or index >= total_samples:
                print(f"Error: Index {index} is out of bounds (0 to {total_samples-1}).")
                return

            # Print Metadata
            print(f"\n--- Sample {index} Overview ---")
            metadata = {}
            for key in f.keys():
                if key != 'colors_webp':
                    val = f[key][index]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    metadata[key] = val
                    print(f"{key:.<20} {val}")

            if show_image:
                # Decode and Display
                raw_bytes = f['colors_webp'][index].tobytes()
                img = Image.open(io.BytesIO(raw_bytes))
                
                plt.figure(num=f"Sample {index}", figsize=(10, 6))
                plt.imshow(img)
                plt.title(f"Index: {index}")
                plt.axis('off')
                plt.show()

    except FileNotFoundError:
        print(f"Error: File '{hdf5_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect hydrofoil HDF5 WebP datasets.")
    parser.add_argument("index", type=int, help="The index of the sample to inspect.")
    parser.add_argument("--file", "-f", default="hydrofoil_webp.hdf5", help="Path to the HDF5 file.")
    parser.add_argument("--no-img", action="store_true", help="Only show metadata, don't open the image.")

    args = parser.parse_args()
    
    inspect_sample(args.file, args.index, show_image=not args.no_img)