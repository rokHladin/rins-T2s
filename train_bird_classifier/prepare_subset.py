import os
import shutil
from PIL import Image
import pandas as pd

# === Step 1: Define target classes (those 24 species) ===
target_classes = {
    '002.Laysan_Albatross',
    '012.Yellow_headed_Blackbird',
    '014.Indigo_Bunting',
    '025.Pelagic_Cormorant',
    '029.American_Crow',
    '033.Yellow_billed_Cuckoo',
    '035.Purple_Finch',
    '042.Vermilion_Flycatcher',
    '048.European_Goldfinch',
    '050.Eared_Grebe',
    '059.California_Gull',
    '068.Ruby_throated_Hummingbird',
    '073.Blue_Jay',
    '081.Pied_Kingfisher',
    '095.Baltimore_Oriole',
    '101.White_Pelican',
    '106.Horned_Puffin',
    '108.White_necked_Raven',
    '112.Great_Grey_Shrike',
    '118.House_Sparrow',
    '134.Cape_Glossy_Starling',
    '138.Tree_Swallow',
    '144.Common_Tern',
    '191.Red_headed_Woodpecker'
}

# === Step 2: Paths ===
CUB_ROOT = "CUB_200_2011/CUB_200_2011"
IMG_DIR = os.path.join(CUB_ROOT, "images")
OUT_DIR = "filtered_data"
os.makedirs(OUT_DIR, exist_ok=True)

# === Step 3: Load metadata ===
images_df = pd.read_csv(os.path.join(CUB_ROOT, "images.txt"), sep=" ", header=None, names=["img_id", "rel_path"])
labels_df = pd.read_csv(os.path.join(CUB_ROOT, "image_class_labels.txt"), sep=" ", header=None, names=["img_id", "class_id"])
bboxes_df = pd.read_csv(os.path.join(CUB_ROOT, "bounding_boxes.txt"), sep=" ", header=None, names=["img_id", "x", "y", "w", "h"])
classes_df = pd.read_csv(os.path.join(CUB_ROOT, "classes.txt"), sep=" ", header=None, names=["class_id", "class_name"])

# === Step 4: Filter for the 24 target species ===
target_class_ids = {
    int(entry.split(".")[0])
    for entry in target_classes
}

# Map class ID to class name without prefix number
class_id_to_name = {
    row.class_id: row.class_name.split(".")[1]
    for _, row in classes_df.iterrows()
    if row.class_id in target_class_ids
}

# === Step 5: Merge all metadata ===
df = images_df.merge(labels_df, on="img_id").merge(bboxes_df, on="img_id")

# Only keep rows from target classes
df = df[df["class_id"].isin(target_class_ids)]

# === Step 6: Crop and save images ===
for _, row in df.iterrows():
    rel_path = row["rel_path"]
    img_path = os.path.join(IMG_DIR, rel_path)
    class_id = row["class_id"]
    class_name = class_id_to_name[class_id]
    out_dir = os.path.join(OUT_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        cropped = img.crop((x, y, x + w, y + h))
        filename = os.path.basename(rel_path)
        cropped.save(os.path.join(out_dir, filename))
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")

print(f"✅ Done! Cropped and filtered dataset saved in: {OUT_DIR}")

