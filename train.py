import os
import time
import logging
import threading
import tempfile
import traceback
import signal
import sys
import shutil
import stat
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax
from torch.optim import AdamW
from torch.cuda import amp
from tqdm import tqdm

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from dataset_loader import LipReadingDataset, collate_fn
from models.frontend.frontend3dcvt import Frontend3DCVT
from models.backend.bigru import BiGRUBackend

# -------------------------
# CONFIG - tune these!
# -------------------------
LABEL_MAP_PATH = "label_map.json"
DATA_TRAIN = "data/splits/grid/train"
DATA_VAL = "data/splits/grid/val"
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
TARGET_SIZE = (64, 64)
FRAME_SKIP = 3

NUM_CPUS = os.cpu_count() or 4
NUM_WORKERS = max(0, min(6, NUM_CPUS - 1))
if sys.platform == "win32":
    NUM_WORKERS = 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

SAVE_INTERVAL_BATCHES = 200
GRAD_CLIP_NORM = 5.0

CHAR_LIST = "abcdefghijklmnopqrstuvwxyz '"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHAR_LIST)}
CHAR2IDX["<blank>"] = 0
NUM_CLASSES = len(CHAR2IDX)

LAST_CKPT = os.path.join(SAVE_DIR, "latest_checkpoint.pth")
BEST_CKPT = os.path.join(SAVE_DIR, "best_checkpoint.pth")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

current_frontend = None
current_backend = None
current_optimizer = None
current_scaler = None
current_epoch = 1
current_best_val_loss = float("inf")

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _has_enough_disk_space(path, min_bytes=50 * 1024 * 1024):
    try:
        statv = shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".")
        return statv.free >= min_bytes
    except Exception:
        return True

def _fsync_file_and_dir(file_path):
    try:
        fd = os.open(file_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass
    try:
        dirpath = os.path.dirname(file_path) or "."
        dirfd = os.open(dirpath, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except Exception:
        pass

def _atomic_save(state: dict, path: str, attempts=2):
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    if not _has_enough_disk_space(path):
        raise RuntimeError(f"Insufficient disk space to save checkpoint: {path}")
    last_exc = None
    for attempt in range(1, attempts + 1):
        with tempfile.NamedTemporaryFile(dir=dirpath, prefix="tmp_ckpt_", delete=False) as tf:
            tmpname = tf.name
        try:
            torch.save(state, tmpname)
            try:
                _fsync_file_and_dir(tmpname)
            except Exception:
                pass
            os.replace(tmpname, path)
            try:
                _fsync_file_and_dir(path)
            except Exception:
                pass
            return
        except Exception as e:
            last_exc = e
            logger.exception(f"Attempt {attempt} failed saving checkpoint to {path}: {e}")
            try:
                if os.path.exists(tmpname):
                    os.remove(tmpname)
            except Exception:
                pass
            time.sleep(0.2 * attempt)
    raise RuntimeError(f"Failed saving checkpoint to {path} after {attempts} attempts") from last_exc

def _make_cpu_state(frontend, backend, optimizer, scaler, epoch, best_val_loss):
    state = {
        "frontend": {k: v.cpu() for k, v in frontend.state_dict().items()},
        "backend": {k: v.cpu() for k, v in backend.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        try:
            state["scaler"] = scaler.state_dict()
        except Exception:
            logger.warning("Failed to include scaler state in checkpoint")
            state["scaler"] = None
    else:
        state["scaler"] = None
    return state

def async_save_checkpoint(frontend, backend, optimizer, scaler, epoch, best_val_loss, path):
    try:
        state = _make_cpu_state(frontend, backend, optimizer, scaler, epoch, best_val_loss)
    except Exception as e:
        logger.warning(f"Failed preparing checkpoint state: {e}")
        return
    def _save_worker(s, p):
        try:
            _atomic_save(s, p)
            logger.debug(f"Async checkpoint saved: {p}")
        except Exception as ex:
            logger.error(f"Async checkpoint save failed: {ex}")
    th = threading.Thread(target=_save_worker, args=(state, path), daemon=True)
    th.start()

def load_checkpoint_if_exists(frontend, backend, optimizer, scaler):
    start_epoch = 1
    best_val_loss = float("inf")
    def load_ckpt(path):
        ckpt = torch.load(path, map_location=DEVICE)
        frontend.load_state_dict(ckpt["frontend"])
        backend.load_state_dict(ckpt["backend"])
        if "optimizer" in ckpt and optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                logger.debug("Could not load optimizer state (version mismatch or missing).")
        if "scaler" in ckpt and scaler is not None and ckpt["scaler"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                logger.debug("Could not load scaler state.")
        epoch = ckpt.get("epoch", 1)
        val_loss = ckpt.get("best_val_loss", float("inf"))
        return epoch, val_loss
    if os.path.isfile(LAST_CKPT):
        try:
            logger.info(f"Found last checkpoint {LAST_CKPT}, loading...")
            start_epoch, best_val_loss = load_ckpt(LAST_CKPT)
            logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        except Exception:
            logger.error(f"Error loading last checkpoint {LAST_CKPT}:\n{traceback.format_exc()}")
    elif os.path.isfile(BEST_CKPT):
        try:
            logger.info(f"Found best checkpoint {BEST_CKPT}, loading weights...")
            start_epoch, best_val_loss = load_ckpt(BEST_CKPT)
            logger.info(f"Resuming from best checkpoint epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        except Exception:
            logger.error(f"Error loading best checkpoint {BEST_CKPT}:\n{traceback.format_exc()}")
    return start_epoch, best_val_loss

def save_checkpoint_and_exit(signum=None, frame=None):
    global current_frontend, current_backend, current_optimizer, current_scaler
    global current_epoch, current_best_val_loss
    logger.info(f"Received termination signal ({signum}). Saving latest checkpoint before exit...")
    try:
        if current_frontend is not None and current_backend is not None and current_optimizer is not None:
            save_checkpoint(
                current_frontend,
                current_backend,
                current_optimizer,
                current_scaler,
                current_epoch,
                current_best_val_loss,
                is_best=False
            )
        else:
            logger.warning("Model or optimizer not initialized yet, skipping checkpoint save.")
    except Exception as e:
        logger.error(f"Error while saving checkpoint during termination: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, save_checkpoint_and_exit)
signal.signal(signal.SIGTERM, save_checkpoint_and_exit)

def infer_frontend_feat_dim(frontend_module, device, try_T=16, C=1, H=64, W=64):
    frontend_module.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, C, try_T, H, W, device=device)
        out = frontend_module(dummy.float())
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim == 3:
            return out.shape[2]
        return out.shape[-1]

import re

def save_checkpoint(frontend, backend, optimizer, scaler, epoch, best_val_loss, is_best=False):
    # Save the main checkpoint for this epoch
    epoch_ckpt = os.path.join(SAVE_DIR, f"epoch_{epoch}.pth")
    state = _make_cpu_state(frontend, backend, optimizer, scaler, epoch, best_val_loss)
    _atomic_save(state, epoch_ckpt)
    # Copy as latest checkpoint (always overwrite)
    _atomic_save(state, LAST_CKPT)
    # Save best checkpoint if needed
    if is_best:
        _atomic_save(state, BEST_CKPT)

    # Delete all batch-level checkpoints created during this epoch (epoch_X_batch_Y.pth)
    batch_ckpt_pattern = re.compile(rf"epoch_{epoch}_batch_\d+\.pth")
    for f in glob.glob(os.path.join(SAVE_DIR, f"epoch_{epoch}_batch_*.pth")):
        try:
            if batch_ckpt_pattern.match(os.path.basename(f)):
                os.remove(f)
                logger.info(f"Deleted batch checkpoint from epoch {epoch}: {f}")
        except Exception as e:
            logger.error(f"Error deleting batch checkpoint {f}: {e}")

    # Now keep only last 3 epoch checkpoints (excluding best)
    # List all epoch checkpoint files but exclude batch files
    all_epoch_ckpts = []
    for f in glob.glob(os.path.join(SAVE_DIR, "epoch_*.pth")):
        base = os.path.basename(f)
        if re.fullmatch(r"epoch_(\d+)\.pth", base):
            all_epoch_ckpts.append(f)

    # Sort by epoch number
    all_epoch_ckpts.sort(key=lambda x: int(re.findall(r"epoch_(\d+)\.pth", os.path.basename(x))[0]))

    # Delete older epoch checkpoints beyond the latest 3
    if len(all_epoch_ckpts) > 3:
        for old_ckpt in all_epoch_ckpts[:-3]:
            # Never delete best checkpoint even if names conflict
            if old_ckpt != BEST_CKPT:
                try:
                    os.remove(old_ckpt)
                    logger.info(f"Deleted old epoch checkpoint: {old_ckpt}")
                except Exception as e:
                    logger.error(f"Error deleting old epoch checkpoint {old_ckpt}: {e}")


def main():
    global current_frontend, current_backend, current_optimizer, current_scaler
    global current_epoch, current_best_val_loss

    set_seed()
    logger.info(f"Device: {DEVICE}, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}, workers={NUM_WORKERS}")

    try:
        train_dataset = LipReadingDataset(DATA_TRAIN, LABEL_MAP_PATH, debug=False, preload=True, target_size=TARGET_SIZE, frame_skip=FRAME_SKIP)
        val_dataset = LipReadingDataset(DATA_VAL, LABEL_MAP_PATH, debug=False, preload=True, target_size=TARGET_SIZE, frame_skip=FRAME_SKIP)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}\n{traceback.format_exc()}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    frontend = Frontend3DCVT().to(DEVICE)
    feat_dim = infer_frontend_feat_dim(frontend, DEVICE, H=TARGET_SIZE[0], W=TARGET_SIZE[1])
    logger.info(f"Inferred frontend feature dim = {feat_dim}")
    backend = BiGRUBackend(feat_dim, 512, 2, NUM_CLASSES, dropout=0.1).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(list(frontend.parameters()) + list(backend.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    start_epoch, best_val_loss = load_checkpoint_if_exists(frontend, backend, optimizer, scaler)

    current_frontend = frontend
    current_backend = backend
    current_optimizer = optimizer
    current_scaler = scaler
    current_epoch = start_epoch
    current_best_val_loss = best_val_loss

    for epoch in range(start_epoch, EPOCHS + 1):
        current_epoch = epoch
        logger.info(f"=== Epoch {epoch}/{EPOCHS} ===")
        t0 = time.time()
        frontend.train()
        backend.train()
        train_loss = 0.0
        batch_idx = 0
        total_batches = len(train_loader)

        pbar = tqdm(total=total_batches, desc=f"Train Epoch {epoch}", unit="batch")
        optimizer.zero_grad()

        for videos, targets, input_lengths, target_lengths in train_loader:
            batch_idx += 1
            retry = 0
            while retry < 3:
                try:
                    videos = videos.permute(0, 2, 1, 3, 4).contiguous().float()
                    videos = videos.to(DEVICE, non_blocking=True)
                    targets = targets.to(DEVICE, non_blocking=True)
                    input_lengths = input_lengths.to(DEVICE)
                    target_lengths = target_lengths.to(DEVICE)

                    with amp.autocast(enabled=torch.cuda.is_available()):
                        feats = frontend(videos)
                        if isinstance(feats, tuple):
                            feats = feats[0]
                        if feats.ndim != 3:
                            feats = feats.view(feats.size(0), feats.size(1), -1)
                        adjusted_input_lengths = (input_lengths.float() * (feats.shape[1] / float(videos.shape[2]))).ceil()
                        adjusted_input_lengths = torch.clamp(adjusted_input_lengths, min=1).to(torch.long)
                        logits = backend(feats)
                        log_probs = log_softmax(logits, dim=-1).permute(1, 0, 2)
                        loss = criterion(log_probs, targets, adjusted_input_lengths.cpu(), target_lengths.cpu())
                        loss = loss / GRAD_ACCUM_STEPS

                    scaler.scale(loss).backward()
                    if batch_idx % GRAD_ACCUM_STEPS == 0 or batch_idx == total_batches:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(list(frontend.parameters()) + list(backend.parameters()), GRAD_CLIP_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    train_loss += loss.item() * GRAD_ACCUM_STEPS
                    pbar.set_postfix({"avg_loss": f"{train_loss / batch_idx:.4f}", "batch": f"{batch_idx}/{total_batches}"})
                    pbar.update(1)

                    if SAVE_INTERVAL_BATCHES > 0 and batch_idx % SAVE_INTERVAL_BATCHES == 0:
                        logger.info(f"Async saving checkpoint: epoch={epoch} batch={batch_idx}")
                        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch}_batch_{batch_idx}.pth")
                        async_save_checkpoint(frontend, backend, optimizer, scaler, epoch, best_val_loss, ckpt_path)

                        # Copy as latest asynchronously:
                        async_save_checkpoint(frontend, backend, optimizer, scaler, epoch, best_val_loss, LAST_CKPT)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        retry += 1
                        logger.error(f"CUDA OOM during training batch {batch_idx}, retry {retry}/3. Clearing cache.")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        if retry == 3:
                            logger.error("OOM: Skipping this batch after 3 retries.")
                            break
                    else:
                        logger.error(f"Runtime error in training batch {batch_idx} (skipping):\n{traceback.format_exc()}")
                        optimizer.zero_grad()
                        break
                except Exception:
                    logger.error(f"Unexpected error in training batch {batch_idx} (skipping):\n{traceback.format_exc()}")
                    optimizer.zero_grad()
                    break
        pbar.close()
        avg_train_loss = train_loss / max(batch_idx, 1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        frontend.eval()
        backend.eval()
        val_loss = 0.0
        val_batches = 0
        pbar_val = tqdm(total=len(val_loader), desc=f"Val Epoch {epoch}", unit="batch")
        with torch.no_grad():
            for videos, targets, input_lengths, target_lengths in val_loader:
                val_batches += 1
                try:
                    videos = videos.permute(0, 2, 1, 3, 4).contiguous().float()
                    videos = videos.to(DEVICE, non_blocking=True)
                    targets = targets.to(DEVICE, non_blocking=True)
                    input_lengths = input_lengths.to(DEVICE)
                    target_lengths = target_lengths.to(DEVICE)
                    with amp.autocast(enabled=torch.cuda.is_available()):
                        feats = frontend(videos)
                        if isinstance(feats, tuple):
                            feats = feats[0]
                        if feats.ndim != 3:
                            feats = feats.view(feats.size(0), feats.size(1), -1)
                        adjusted_input_lengths = (input_lengths.float() * (feats.shape[1] / float(videos.shape[2]))).ceil()
                        adjusted_input_lengths = torch.clamp(adjusted_input_lengths, min=1).to(torch.long)
                        logits = backend(feats)
                        log_probs = log_softmax(logits, dim=-1).permute(1, 0, 2)
                        loss = criterion(log_probs, targets, adjusted_input_lengths.cpu(), target_lengths.cpu())
                        val_loss += float(loss.item())
                    pbar_val.set_postfix({"avg_loss": f"{val_loss / val_batches:.4f}"})
                    pbar_val.update(1)
                except Exception:
                    logger.error(f"Error in validation batch {val_batches} (skipping):\n{traceback.format_exc()}")
                    continue
        pbar_val.close()
        avg_val_loss = val_loss / max(val_batches, 1)
        logger.info(f"Saving checkpoint for epoch {epoch}...")

        # Save as epoch_X.pth and as latest.pth, and purge old epoch checkpoints
        try:
            save_checkpoint(frontend, backend, optimizer, scaler, epoch, best_val_loss, is_best=False)
        except Exception:
            logger.error(f"Error saving checkpoint:\n{traceback.format_exc()}")

        if avg_val_loss < best_val_loss:
            logger.info(f"Validation improved: {avg_val_loss:.4f} < previous_best {best_val_loss:.4f}. Saving best -> {BEST_CKPT}")
            best_val_loss = avg_val_loss
            current_best_val_loss = best_val_loss
            try:
                save_checkpoint(frontend, backend, optimizer, scaler, epoch, best_val_loss, is_best=True)
            except Exception:
                logger.error(f"Error saving best checkpoint:\n{traceback.format_exc()}")

        logger.info(f"Epoch {epoch}/{EPOCHS} - TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} Time={time.time() - t0:.1f}s")

    logger.info("Training complete.")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory error! Try reducing BATCH_SIZE or increasing FRAME_SKIP.")
        else:
            logger.error(f"Runtime error:\n{traceback.format_exc()}")
        save_checkpoint_and_exit()
    except Exception:
        logger.error(f"Unexpected error:\n{traceback.format_exc()}")
        save_checkpoint_and_exit()
