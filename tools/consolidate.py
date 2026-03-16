import h5py
import os
import glob
import numpy as np
import io
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
source_folder = './output'
output_file = 'hydrofoil_webp.hdf5'
webp_quality = 80  # 80 is standard; 100 is very high quality; <50 is very small

# 1. Gather files and calculate original size programmatically
files = sorted(glob.glob(os.path.join(source_folder, "*.hdf5")))
original_total_bytes = sum(os.path.getsize(f) for f in files)

# 2. Pre-scan for scalars/strings (assuming all files have same keys)
with h5py.File(files[0], 'r') as f:
    other_keys = [k for k in f.keys() if k != 'colors']
    
num_samples = len(files)

# 3. Process and Compress
with h5py.File(output_file, 'w') as master_h5:
    # Create Variable-Length type for encoded WebP bytes
    vlen_type = h5py.special_dtype(vlen=np.dtype('uint8'))
    img_ds = master_h5.create_dataset('colors_webp', (num_samples,), dtype=vlen_type)
    
    # Setup scalar/string datasets
    scalar_ds = {k: master_h5.create_dataset(k, (num_samples,), dtype=h5py.special_dtype(vlen=str) if 'source' in k else 'f') for k in other_keys}

    for i, file_path in enumerate(tqdm(files, desc="WebP Encoding", unit="file", colour='cyan')):
        try:
            with h5py.File(file_path, 'r') as src:
                # Encode Image to WebP
                img_array = src['colors'][...]
                img = Image.fromarray(img_array)
                
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=webp_quality)
                
                # Save binary blob
                img_ds[i] = np.frombuffer(buf.getvalue(), dtype='uint8')
                
                # Save metadata/scalars
                for k in other_keys:
                    val = src[k][()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    scalar_ds[k][i] = val
        except Exception as e:
            print(f"\nError on {file_path}: {e}")

# 4. Final Comparison Results
final_size_bytes = os.path.getsize(output_file)
reduction = (1 - (final_size_bytes / original_total_bytes)) * 100

print(f"\n--- Final Comparison ---")
print(f"Original Folder Size: {original_total_bytes / (1024**3):.2f} GB")
print(f"New HDF5 (WebP) Size: {final_size_bytes / (1024**3):.2f} GB")
print(f"Net Space Saved:      {reduction:.2f}%")