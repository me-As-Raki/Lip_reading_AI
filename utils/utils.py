import torch
import os
import json


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints", filename="last.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"[CHECKPOINT] Saved latest model to {path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)
        print(f"[CHECKPOINT] Saved BEST model to {best_path}")


def load_label_map(path="label_map.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map file not found: {path}")
    with open(path, "r") as f:
        label_map = json.load(f)
    print(f"[LABEL MAP] Loaded label map from {path}")
    return label_map


def decode_prediction(pred_indices, inv_label_map):
    """
    Converts predicted indices into string.
    Assumes CTC output: remove repeats and blanks.
    """
    pred_str = ""
    prev = -1
    for i in pred_indices:
        if i != prev and i != 0:
            pred_str += inv_label_map[str(i)]
        prev = i
    return pred_str
 
