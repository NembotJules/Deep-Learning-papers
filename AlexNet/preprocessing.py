import os
import shutil
from pathlib import Path

parent_folder = "tiny-imagenet-200/train"

wnist_dict = {}

for i, folder in enumerate(os.listdir(parent_folder)): 
    wnist_dict[folder] = f"{i}"

print(wnist_dict['n02509815'])


# Step 1: Define paths
base_dir = Path("tiny-imagenet-200/val")
images_dir = base_dir / "images"
annotations_file = base_dir / "val_annotations.txt"

# Step 2: Read wnids.txt to build WNID-to-index mapping (assuming it's in train/)
# You already have this, but I'll include it for completeness
wnids_file = Path("tiny-imagenet-200/wnids.txt")
with open(wnids_file, 'r') as f:
    wnids = [line.strip() for line in f.readlines()]
wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

# Step 3: Create class folders and move images
with open(annotations_file, 'r') as f:
    for line in f:
        # Parse the line: "val_0.JPEG n03444034 0 32 44 62"
        parts = line.strip().split()
        image_name = parts[0]  # e.g., "val_0.JPEG"
        wnid = parts[1]       # e.g., "n03444034"

        # Source path for the image
        src_path = images_dir / image_name

        # Destination folder (create if it doesn't exist)
        dest_dir = base_dir / wnid
        dest_dir.mkdir(exist_ok=True)

        # Destination path for the image
        dest_path = dest_dir / image_name

        # Move the image to the class folder
        shutil.move(src_path, dest_path)

# Step 4: Remove the empty 'images' folder (optional, if it's now empty)
try:
    images_dir.rmdir()
    print(f"Removed empty 'images' folder at {images_dir}")
except OSError as e:
    print(f"Could not remove 'images' folder: {e}. It may not be empty or there are permission issues.")

print("Validation set reorganized into class folders.")