import os
import json

# Folders to scan
split_dirs = [
    "data/splits/grid/train",
    "data/splits/grid/val",
    "data/splits/grid/test"
]

label_map = {}
word_index = 1

for split_dir in split_dirs:
    if not os.path.exists(split_dir):
        print(f"⚠️ Directory not found: {split_dir}")
        continue

    for fname in os.listdir(split_dir):
        if fname.endswith(".npy"):
            key = fname[:-4]  # remove .npy
            if key not in label_map:
                label_map[key] = f"word{word_index}"
                word_index += 1

# Save to label_map.json
os.makedirs("data", exist_ok=True)
with open("data/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"✅ Created data/label_map.json with {len(label_map)} entries.")

