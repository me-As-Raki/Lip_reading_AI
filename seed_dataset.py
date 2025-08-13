# seed_dataset.py
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ==== CONFIG ====
DATASET_SPLIT_DIR = "data/splits/grid/train"   # change to val/test when needed
LABEL_MAP_PATH = "label_map.json"
OUTPUT_DIR = "data/preprocessed/grid/train_pt"  # where preprocessed .pt will be saved

TARGET_SIZE = (64, 64)   # resize target (H, W)
FRAME_SKIP = 2           # use every nth frame
USE_GRAYSCALE = True
FP16 = True

# ==== CHAR MAP ====
CHAR_LIST = "abcdefghijklmnopqrstuvwxyz '"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHAR_LIST)}
CHAR2IDX["<blank>"] = 0


def load_and_process_npy(path: str) -> torch.Tensor:
    import os
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Try loading as a normal numpy array without pickle
        data = np.load(path, allow_pickle=False)
    except Exception as e_np:
        print(f"[WARN] np.load failed with allow_pickle=False for {path}: {e_np}")
        try:
            # Fallback: try with allow_pickle=True if previous fails
            data = np.load(path, allow_pickle=True)
        except Exception as e_pkl:
            raise RuntimeError(f"Failed to load numpy file {path} with pickle: {e_pkl}")

    if isinstance(data, np.ndarray) and data.dtype == object:
        # For object arrays, try to extract 'video' key if it's a dict
        data_item = data.item()
        if isinstance(data_item, dict) and "video" in data_item:
            video_np = data_item["video"]
        else:
            raise ValueError(f"Unexpected npy object format in {path}")
    elif isinstance(data, np.ndarray):
        video_np = data
    else:
        raise ValueError(f"Unexpected npy content in {path}")

    # Check dimensions and reshape if needed
    if video_np.ndim == 3:
        T, H, W = video_np.shape
        video_np = video_np.reshape((T, 1, H, W))
    elif video_np.ndim != 4:
        raise ValueError(f"Unexpected video shape {video_np.shape} in {path}")

    # Convert to grayscale if needed
    if USE_GRAYSCALE and video_np.shape[1] == 3:
        video_np = np.mean(video_np, axis=1, keepdims=True)

    # Frame skip
    if FRAME_SKIP > 1:
        video_np = video_np[::FRAME_SKIP]

    # Convert to tensor
    video_tensor = torch.from_numpy(video_np).float()

    # Resize
    if TARGET_SIZE is not None:
        H, W = TARGET_SIZE
        video_tensor = F.interpolate(video_tensor, size=(H, W), mode='bilinear', align_corners=False)

    # FP16
    if FP16:
        video_tensor = video_tensor.half()

    return video_tensor


def encode_label(label: str) -> torch.Tensor:
    return torch.tensor([CHAR2IDX.get(ch, 0) for ch in label.lower()], dtype=torch.long)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load label map
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    root_dir_norm = os.path.normpath(DATASET_SPLIT_DIR)
    split_name = os.path.basename(root_dir_norm)

    index_entries = []

    for speaker in sorted(os.listdir(DATASET_SPLIT_DIR)):
        spk_path = os.path.join(DATASET_SPLIT_DIR, speaker)
        if not os.path.isdir(spk_path):
            continue

        for file in sorted(os.listdir(spk_path)):
            if not file.endswith(".npy"):
                continue

            filename_wo_ext = os.path.splitext(file)[0]
            key = f"{split_name}/{speaker}/{filename_wo_ext}"

            if key not in label_map:
                continue

            video_path = os.path.join(spk_path, file)
            video_tensor = load_and_process_npy(video_path)
            label_tensor = encode_label(label_map[key])

            out_path = os.path.join(OUTPUT_DIR, f"{speaker}_{filename_wo_ext}.pt")
            torch.save({"video": video_tensor, "label": label_tensor}, out_path)

            index_entries.append(out_path)

    # Save index file
    index_path = os.path.join(OUTPUT_DIR, "index.txt")
    with open(index_path, "w") as f:
        for p in index_entries:
            f.write(p + "\n")

    print(f"[INFO] Preprocessing complete. {len(index_entries)} samples saved in {OUTPUT_DIR}")
    print(f"[INFO] Index file saved to {index_path}")


if __name__ == "__main__":
    main()
