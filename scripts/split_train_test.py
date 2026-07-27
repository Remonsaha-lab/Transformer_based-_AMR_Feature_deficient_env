import os
import random
import shutil
from pathlib import Path

src_dir = Path("data/diode/raw_ply_data/train")
dest_dir = Path("data/diode/raw_ply_data/test")
random.seed(42)

# Find all .ply files recursively
all_plys = sorted(src_dir.rglob("*.ply"))
print(f"Found {len(all_plys)} total point clouds under {src_dir}")

# Select 10% of files
num_to_move = int(len(all_plys) * 0.10)
print(f"Moving {num_to_move} point clouds to {dest_dir}...")

test_plys = random.sample(all_plys, num_to_move)
moved_count = 0

for ply_path in test_plys:
    # Determine the corresponding label path
    label_path = Path(str(ply_path).replace(".ply", "_labels.npy"))
    
    files_to_move = [ply_path]
    if label_path.exists():
        files_to_move.append(label_path)
        
    for f in files_to_move:
        # Get path relative to the train folder
        rel_path = f.relative_to(src_dir)
        dest_file = dest_dir / rel_path
        
        # Recreate the folder structure in the test folder
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(f), str(dest_file))
        
    moved_count += 1

print(f"Successfully moved {moved_count} point clouds and their label files to {dest_dir}")
