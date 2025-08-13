import os
import json
import re

# === CONFIGURATION ===
PROCESSED_BASE = "data/splits/grid"  # base path for processed npy files (train, val, test folders)
RAW_BASE = "data/raw/grid"            # base path for raw .align files (speaker folders without _processed)
LABEL_MAP_PATH = "label_map.json"

# Silence tokens to skip in align files
SILENCE_TOKENS = {"sil", "sp"}

def parse_align_file(align_path):
    """
    Parse .align file and extract words ignoring silences.
    .align format assumed: <start> <end> <word>
    """
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, _, word = parts
            if word.lower() not in SILENCE_TOKENS:
                words.append(word.lower())
    return " ".join(words)

def generate_label_map():
    label_map = {}
    missing_labels = 0
    total_files = 0

    # Expected data splits: train, val, test
    splits = ['train', 'val', 'test']

    print(f"[INFO] Starting label map generation...")

    for split in splits:
        split_path = os.path.join(PROCESSED_BASE, split)
        if not os.path.isdir(split_path):
            print(f"[WARNING] Split folder not found: {split_path}, skipping.")
            continue

        # List all speaker folders under split (e.g. s1_processed)
        speakers = sorted(os.listdir(split_path))
        print(f"[INFO] Processing split '{split}' with {len(speakers)} speakers.")

        for speaker in speakers:
            speaker_processed_path = os.path.join(split_path, speaker)
            # Convert speaker_processed 's1_processed' -> raw 's1'
            speaker_raw_name = speaker.replace('_processed', '')
            speaker_raw_path = os.path.join(RAW_BASE, speaker_raw_name, "align")

            if not os.path.isdir(speaker_processed_path):
                print(f"[WARNING] Processed path missing: {speaker_processed_path}, skipping.")
                continue
            if not os.path.isdir(speaker_raw_path):
                print(f"[WARNING] Raw align path missing: {speaker_raw_path}, skipping.")
                continue

            # Get all .npy files for speaker
            npy_files = [f for f in os.listdir(speaker_processed_path) if f.endswith(".npy")]
            print(f"[INFO] Speaker '{speaker}': Found {len(npy_files)} npy files.")

            for npy_file in npy_files:
                total_files += 1
                file_id = os.path.splitext(npy_file)[0]  # e.g. 'bbaf2n'

                # Corresponding align file path
                align_file = os.path.join(speaker_raw_path, file_id + ".align")

                if not os.path.isfile(align_file):
                    print(f"[WARNING] Align file missing for {file_id} at {align_file}, skipping.")
                    missing_labels += 1
                    continue

                # Parse .align file to extract label sentence
                label = parse_align_file(align_file)
                if len(label.strip()) == 0:
                    print(f"[WARNING] Empty label extracted for {file_id}, skipping.")
                    missing_labels += 1
                    continue

                # Store label with full relative key: split/speaker_processed/filename (optional)
                key = f"{split}/{speaker}/{file_id}"
                label_map[key] = label

    print(f"[INFO] Label map generation completed.")
    print(f"[INFO] Total .npy files found: {total_files}")
    print(f"[INFO] Successfully labeled files: {len(label_map)}")
    print(f"[INFO] Missing or invalid labels skipped: {missing_labels}")

    # Save label map JSON
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[INFO] Label map saved to '{LABEL_MAP_PATH}'")

if __name__ == "__main__":
    generate_label_map()
