import os
import shutil
import re

SRC_FOLDER = 'images_to_duplicate'   # Where your special images are
DST_ROOT = 'filtered_data'           # Where your filtered dataset lives
N_COPIES = 50                        # Number of copies per image

def get_species_name(filename):
    # e.g. "american_crow.jpg" -> "american_crow"
    return os.path.splitext(filename)[0].lower()

def get_next_index(dst_dir, species):
    """Return the next available index for files like species_XXX.jpg in dst_dir."""
    pattern = re.compile(rf"{re.escape(species)}_(\d+)\.jpg", re.IGNORECASE)
    existing = [
        int(m.group(1))
        for fname in os.listdir(dst_dir)
        if (m := pattern.fullmatch(fname))
    ]
    return max(existing, default=0) + 1

for fname in os.listdir(SRC_FOLDER):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        species = get_species_name(fname)
        dst_dir = os.path.join(DST_ROOT, species)
        os.makedirs(dst_dir, exist_ok=True)

        src_path = os.path.join(SRC_FOLDER, fname)
        start_idx = get_next_index(dst_dir, species)
        for i in range(N_COPIES):
            dst_fname = f"{species}_{start_idx + i:03d}.jpg"
            dst_path = os.path.join(dst_dir, dst_fname)
            shutil.copy(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")

print("✅ Done! No overwriting occurred.")

