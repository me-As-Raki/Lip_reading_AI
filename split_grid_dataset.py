import os
import shutil
import random
from datetime import datetime

def log(msg):
    """Timestamped logging to console."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def split_dataset(
    processed_path="data/processed/grid",
    splits_base="data/splits/grid",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    log("Starting dataset split...")

    # Validate ratios sum
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1. Got {total_ratio}")

    # Create split directories
    train_dir = os.path.join(splits_base, "train")
    val_dir = os.path.join(splits_base, "val")
    test_dir = os.path.join(splits_base, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        log(f"Ensured directory exists: {d}")

    # Seed for reproducibility
    random.seed(seed)

    speakers = sorted(
        [d for d in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, d))]
    )
    log(f"Found {len(speakers)} speakers in processed data.")

    total_files = 0
    total_train, total_val, total_test = 0, 0, 0

    for spk in speakers:
        spk_path = os.path.join(processed_path, spk)
        files = [f for f in os.listdir(spk_path) if f.endswith(".npy")]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        def copy_files(file_list, dest_base):
            dest_spk_dir = os.path.join(dest_base, spk)
            os.makedirs(dest_spk_dir, exist_ok=True)
            for f in file_list:
                src = os.path.join(spk_path, f)
                dst = os.path.join(dest_spk_dir, f)
                shutil.copy2(src, dst)

        copy_files(train_files, train_dir)
        copy_files(val_files, val_dir)
        copy_files(test_files, test_dir)

        log(
            f"Speaker {spk}: Total {n} videos | "
            f"Train {len(train_files)} | Val {len(val_files)} | Test {len(test_files)}"
        )

        total_files += n
        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

    log(f"Dataset splitting finished.")
    log(f"Total files: {total_files}")
    log(f"Train files: {total_train}")
    log(f"Validation files: {total_val}")
    log(f"Test files: {total_test}")

if __name__ == "__main__":
    split_dataset()
