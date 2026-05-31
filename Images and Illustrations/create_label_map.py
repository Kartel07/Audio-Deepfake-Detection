# create_label_map.py
import os
import json

DATASET_ROOT = os.path.join("Datasets", "for-2sec")
label_map = {}

print("📖 Mapping original files to labels...")
# Walk through the original dataset folders to find the files in their context
for root, _, files in os.walk(DATASET_ROOT):
    for f in files:
        if f.endswith(('.wav', '.flac')):
            # Check the parent folder name to determine the label
            if "fake" in root.lower():
                label_map[f] = 1
            elif "real" in root.lower():
                label_map[f] = 0

with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print(f"✅ Map created with {len(label_map)} entries. Saved to label_map.json")