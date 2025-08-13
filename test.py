import os
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import LipReadingDataset, collate_fn
from models.frontend.frontend3dcvt import Frontend3DCVT
from models.backend.bigru import BiGRUBackend

# ---------- Config ----------
LABEL_MAP_PATH = "label_map.json"
DATA_TEST = "data/splits/grid/test"
SAVE_DIR = "checkpoints"
CKPT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
TARGET_SIZE = (64, 64)
FRAME_SKIP = 3
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHAR_LIST = "abcdefghijklmnopqrstuvwxyz '"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHAR_LIST)}
CHAR2IDX["<blank>"] = 0
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}

# ---------- CER & WER ----------
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions  = current_row[j] + 1
            subs = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, subs))
        previous_row = current_row
    return previous_row[-1]

def cer(reference, prediction):
    if len(reference) == 0: return 0.0 if len(prediction) == 0 else 1.0
    return levenshtein_distance(reference, prediction) / len(reference)

def wer(reference, prediction):
    ref_words = reference.split()
    pred_words = prediction.split()
    if len(ref_words) == 0: return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein_distance(ref_words, pred_words) / len(ref_words)

# ---------- CTC Decoding ----------
def ctc_greedy_decode(log_probs, input_lengths):
    max_probs = torch.argmax(log_probs, dim=2)  # (T, B)
    max_probs = max_probs.cpu().numpy()
    results = []
    for b in range(max_probs.shape[1]):
        prev = -1
        seq = []
        for t in range(int(input_lengths[b])):
            idx = int(max_probs[t, b])
            if idx != prev and idx != 0:
                seq.append(IDX2CHAR.get(idx, ''))
            prev = idx
        results.append("".join(seq))
    return results

# ---------- Load Model ----------
def load_model_from_ckpt(ckpt_path):
    frontend = Frontend3DCVT().to(DEVICE)
    dummy = torch.zeros(1, 1, 16, TARGET_SIZE[0], TARGET_SIZE[1], device=DEVICE)
    feat_dim = frontend(dummy).shape[-1]
    backend = BiGRUBackend(feat_dim, 512, 2, len(CHAR2IDX), dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(list(frontend.parameters()) + list(backend.parameters()))
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    frontend.load_state_dict(ckpt["frontend"])
    backend.load_state_dict(ckpt["backend"])
    print(f"✅ Loaded checkpoint from {ckpt_path}, epoch={ckpt.get('epoch', '?')}, best_val_loss={ckpt.get('best_val_loss', '?')}")
    return frontend, backend

# ---------- Evaluation ----------
def evaluate():
    test_dataset = LipReadingDataset(DATA_TEST, LABEL_MAP_PATH, debug=False, preload=True,
                                     target_size=TARGET_SIZE, frame_skip=FRAME_SKIP)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True, collate_fn=collate_fn)
    frontend, backend = load_model_from_ckpt(CKPT_PATH)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    frontend.eval()
    backend.eval()
    total_loss = 0.0
    total_chars = correct_chars = total_words = correct_words = 0
    total_cer = total_wer = num_samples = 0

    with torch.no_grad():
        for videos, targets, input_lengths, target_lengths in tqdm(test_loader, desc="Testing"):
            videos = videos.permute(0, 2, 1, 3, 4).float().to(DEVICE)
            targets = targets.to(DEVICE)
            input_lengths = input_lengths.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)
            with amp.autocast(enabled=torch.cuda.is_available()):
                feats = frontend(videos)
                if isinstance(feats, tuple): feats = feats[0]
                if feats.ndim != 3: feats = feats.view(feats.size(0), feats.size(1), -1)
                adjusted_input_lengths = (input_lengths.float() * (feats.shape[1] / float(videos.shape[2]))).ceil()
                adjusted_input_lengths = torch.clamp(adjusted_input_lengths, min=1).long()
                logits = backend(feats)
                log_probs = log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss = criterion(log_probs, targets, adjusted_input_lengths.cpu(), target_lengths.cpu())
            total_loss += loss.item() * videos.size(0)
            num_samples += videos.size(0)
            pred_texts = ctc_greedy_decode(log_probs, adjusted_input_lengths.cpu().numpy())
            gt_texts = []
            idx = 0
            for l in target_lengths.cpu().numpy():
                token_seq = targets[idx:idx+l]
                if token_seq.dim() > 1: token_seq = token_seq.view(-1)
                token_ids = token_seq.cpu().tolist()
                gt_str = "".join([IDX2CHAR.get(t, '') for t in token_ids])
                gt_texts.append(gt_str)
                idx += l
            for gt, pred in zip(gt_texts, pred_texts):
                total_chars += len(gt)
                correct_chars += sum(1 for a, b in zip(gt, pred) if a == b)
                total_words += 1
                if gt == pred: correct_words += 1
                total_cer += cer(gt, pred)
                total_wer += wer(gt, pred)
    avg_loss = total_loss / num_samples
    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    word_acc = (correct_words / total_words) * 100 if total_words > 0 else 0
    avg_cer = (total_cer / total_words) * 100 if total_words > 0 else 0
    avg_wer = (total_wer / total_words) * 100 if total_words > 0 else 0
    print(f"\n📊 Test Results:")
    print(f" - Average CTC Loss: {avg_loss:.4f}")
    print(f" - Character Accuracy: {char_acc:.2f}%")
    print(f" - Word Accuracy: {word_acc:.2f}%")
    print(f" - Average Character Error Rate (CER): {avg_cer:.2f}%")
    print(f" - Average Word Error Rate (WER): {avg_wer:.2f}%")

if __name__ == "__main__":
    evaluate()
