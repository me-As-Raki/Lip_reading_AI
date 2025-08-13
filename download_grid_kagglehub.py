import os
import shutil
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Source: where KaggleHub downloaded it
source_dir = os.path.expanduser(
    "~/.cache/kagglehub/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data"
)

# Destination: your project folder
dest_dir = os.path.join("data", "raw", "grid")
os.makedirs(dest_dir, exist_ok=True)

log("Starting to copy all speaker folders...")

for folder in os.listdir(source_dir):
    if folder.endswith("_processed"):
        src = os.path.join(source_dir, folder)
        dst = os.path.join(dest_dir, folder.replace("_processed", ""))
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            log(f"✅ Copied {folder} to {dst}")
        else:
            log(f"⚠️ Skipped {folder}, already exists.")

log("✅ All speaker folders copied successfully.")
