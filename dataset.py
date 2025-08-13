 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from glob import glob


class LipDataset(Dataset):
    def __init__(self, root_dir, split="train", label_map=None, max_len=75):
        """
        root_dir: dataset root (e.g. data/processed/grid)
        split: 'train', 'val', 'test'
        label_map: dict {word: idx}
        max_len: maximum number of frames
        """
        self.split = split
        self.max_len = max_len
        self.label_map = label_map

        self.data_dir = os.path.join(root_dir, split)
        self.samples = sorted(glob(os.path.join(self.data_dir, "*.npy")))
        assert len(self.samples) > 0, f"No samples found in {self.data_dir}"

        print(f"[DATASET] Loaded {len(self.samples)} samples for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        data = np.load(sample_path, allow_pickle=True).item()

        video = data['video']  # (T, H, W) or (T, C, H, W)
        label_str = data['label']  # Word string

        if video.ndim == 3:
            video = np.expand_dims(video, axis=1)  # (T, 1, H, W)

        if self.label_map:
            label_ids = [self.label_map[c] for c in label_str]
        else:
            raise ValueError("Label map not provided!")

        # Convert to tensor
        video_tensor = torch.tensor(video, dtype=torch.float32)  # (T, 1, 96, 96)
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        return video_tensor, label_tensor


def collate_fn(batch):
    """
    Pads batch of variable-length videos and labels
    Returns: videos_padded, labels_padded, input_lengths, label_lengths
    """
    videos, labels = zip(*batch)

    batch_size = len(videos)
    max_video_len = max([v.shape[0] for v in videos])
    max_label_len = max([len(l) for l in labels])

    # Pad video
    video_tensor = torch.zeros((batch_size, max_video_len, 1, 96, 96), dtype=torch.float32)
    input_lengths = []

    for i, v in enumerate(videos):
        video_tensor[i, :v.shape[0]] = v
        input_lengths.append(v.shape[0])

    # Pad labels
    label_tensor = torch.zeros((batch_size, max_label_len), dtype=torch.long)
    label_lengths = []

    for i, l in enumerate(labels):
        label_tensor[i, :len(l)] = l
        label_lengths.append(len(l))

    return video_tensor, label_tensor, torch.tensor(input_lengths), torch.tensor(label_lengths)
