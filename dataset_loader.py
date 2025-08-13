import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple
from tqdm import tqdm


class LipReadingDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        label_map_path: str,
        transform: Optional[callable] = None,
        debug: bool = False,
        target_size: Tuple[int, int] = (64, 64),
        frame_skip: int = 2,
        preload: bool = True,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.debug = debug
        self.target_size = target_size
        self.frame_skip = frame_skip
        self.preload = preload

        # Load label map
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        self.samples = []
        split_name = os.path.basename(os.path.normpath(root_dir))

        # Collect all sample paths that have labels
        for speaker in sorted(os.listdir(root_dir)):
            spk_path = os.path.join(root_dir, speaker)
            if not os.path.isdir(spk_path):
                continue
            for file in sorted(os.listdir(spk_path)):
                if file.endswith(".npy"):
                    filename_wo_ext = os.path.splitext(file)[0]
                    key = f"{split_name}/{speaker}/{filename_wo_ext}"
                    if key in self.label_map:
                        self.samples.append((os.path.join(spk_path, file), self.label_map[key]))

        if self.debug:
            print(f"[INFO] Found {len(self.samples)} labeled samples in {root_dir}")

        # Char mapping
        self.char_list = "abcdefghijklmnopqrstuvwxyz '"
        self.char2idx = {c: i + 1 for i, c in enumerate(self.char_list)}
        self.char2idx["<blank>"] = 0

        self.data_cache = []
        if self.preload:
            print(f"[INFO] Preloading {len(self.samples)} videos into CPU RAM...")
            for file_path, label in tqdm(self.samples, disable=not self.debug):
                try:
                    video_tensor = self._load_and_process(file_path)
                    if video_tensor is None:
                        continue
                    label_tensor = self._encode_label(label)
                    self.data_cache.append((video_tensor, label_tensor))
                except Exception as e:
                    print(f"[WARN] Skipping {file_path}: {e}")

            if self.debug:
                print(f"[INFO] Successfully preloaded {len(self.data_cache)} / {len(self.samples)} samples.")

    def _load_and_process(self, path: str) -> Optional[torch.Tensor]:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")
            return None

        # Handle dict/object format
        if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
            content = data.item()
            video_np = content.get("video")
            if video_np is None:
                return None
        elif isinstance(data, np.ndarray):
            video_np = data
        else:
            return None

        # Ensure at least 3D
        if video_np.ndim == 3:  # (T, H, W)
            T, H, W = video_np.shape
            video_np = video_np.reshape((T, 1, H, W))
        elif video_np.ndim == 4:
            pass  # already (T, C, H, W)
        else:
            return None

        # Convert RGB → grayscale if needed
        if video_np.shape[1] == 3:
            video_np = np.mean(video_np, axis=1, keepdims=True)

        # Sanity check shape
        if video_np.shape[1] != 1:
            return None

        # Frame skipping
        if self.frame_skip > 1:
            video_np = video_np[::self.frame_skip]

        video_tensor = torch.from_numpy(video_np).float()

        # Resize
        if self.target_size is not None:
            H, W = self.target_size
            T, C, H_orig, W_orig = video_tensor.shape
            video_tensor = video_tensor.reshape(T * C, 1, H_orig, W_orig)
            video_tensor = F.interpolate(video_tensor, size=(H, W), mode="bilinear", align_corners=False)
            video_tensor = video_tensor.reshape(T, C, H, W)

        return video_tensor.half()

    def _encode_label(self, label: str) -> torch.Tensor:
        return torch.tensor([self.char2idx.get(ch, 0) for ch in label.lower()], dtype=torch.long)

    def __len__(self):
        return len(self.data_cache) if self.preload else len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            return self.data_cache[idx]
        file_path, label = self.samples[idx]
        video_tensor = self._load_and_process(file_path)
        label_tensor = self._encode_label(label)
        return video_tensor, label_tensor


def collate_fn(batch):
    videos, targets = zip(*batch)
    max_len = max(v.shape[0] for v in videos)

    # Pad videos
    padded_videos = []
    for v in videos:
        if v.shape[0] < max_len:
            padding = torch.zeros((max_len - v.shape[0], v.shape[1], v.shape[2], v.shape[3]),
                                  dtype=v.dtype, device=v.device)
            v = torch.cat([v, padding], dim=0)
        padded_videos.append(v)

    videos_padded = torch.stack(padded_videos, dim=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    input_lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    return videos_padded, targets_padded, input_lengths, target_lengths


if __name__ == "__main__":
    LABEL_MAP_PATH = "label_map.json"
    DATASET_PATH = "data/splits/grid/train"

    dataset = LipReadingDataset(
        DATASET_PATH,
        LABEL_MAP_PATH,
        debug=True,
        preload=True,
        target_size=(64, 64),
        frame_skip=2,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(dataset)}")

    for i, (videos, labels, input_lens, target_lens) in enumerate(loader):
        print(f"Batch {i} → videos: {videos.shape}, labels: {labels.shape}")
        if i >= 2:
            break
