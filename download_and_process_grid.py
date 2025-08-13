"""
download_and_process_grid_gpu_fast.py

Advanced GRID preprocessing:
 - Uses dlib CNN detector on GPU (if available)
 - Multiprocessing where each worker initializes dlib models once
 - Detect every N frames, track between detections (KCF)
 - Saves per-video .npy with {"video": arr, "label": label}
 - Logs timings to processing_times.csv and skipped_files.log
 - Shows overall tqdm progress bar and per-speaker time summary
"""

import os
import cv2
import numpy as np
import dlib
import json
import time
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import csv

# ---------------- CONFIG ----------------
GRID_DATASET_PATH = r"C:\Users\Rakesh\.cache\kagglehub\datasets\jedidiahangekouakou\grid-corpus-dataset-for-training-lipnet\versions\1\data"
PROCESSED_PATH = "data/processed/grid/"
DLIB_PREDICTOR_PATH = "utils/shape_predictor_68_face_landmarks.dat"
MMOD_MODEL_PATH = "utils/mmod_human_face_detector.dat"  # GPU CNN detector file
VIDEO_EXT = ".mpg"
ALIGN_EXT = ".align"
MOUTH_SIZE = 96
SKIP_EXISTING = True
LOG_FILE = "skipped_files.log"
TIMES_CSV = "processing_times.csv"

NUM_WORKERS = max(1, cpu_count() - 1)
DETECT_EVERY_N_FRAMES = 3   # detect on every 3rd frame, track otherwise (tune: 1..5)
MAX_FRAMES_PER_VIDEO = None # set to limit frames for testing, or None for full

# ---------------- GLOBALS (initialized per-worker) ----------------
detector = None
predictor = None
gpu_available = False


# ---------------- Utilities ----------------
def read_label_from_align(speaker_path, filename):
    align_path = os.path.join(speaker_path, "align", filename.replace(VIDEO_EXT, ALIGN_EXT))
    if not os.path.exists(align_path):
        return None
    try:
        with open(align_path, "r", encoding="utf-8") as f:
            words = [line.strip().split()[-1] for line in f if line.strip()]
        words = [w for w in words if w.lower() != "sil"]
        return " ".join(words) if words else None
    except Exception:
        return None


def rect_to_bbox(rect):
    """Convert dlib rect to (x, y, w, h) for trackers & cropping"""
    # rect may be dlib.rectangle or object with left(), top(), right(), bottom()
    try:
        left = int(rect.left())
        top = int(rect.top())
        right = int(rect.right())
        bottom = int(rect.bottom())
    except Exception:
        # if it's a cv-style tuple already
        x, y, w, h = rect
        return x, y, w, h
    w = right - left
    h = bottom - top
    return left, top, w, h


def crop_and_resize_gray(frame, bbox, size=MOUTH_SIZE, margin=10):
    x, y, w, h = bbox
    h_img, w_img = frame.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w_img, x + w + margin)
    y1 = min(h_img, y + h + margin)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mouth = cv2.resize(crop, (size, size))
    return mouth


# ---------------- Worker initializer ----------------
def init_worker(mmod_model_path, predictor_path, detect_gpu_flag):
    """
    Called once per worker process at start.
    Loads dlib models into global variables.
    """
    global detector, predictor, gpu_available
    gpu_available = detect_gpu_flag and hasattr(dlib, "DLIB_USE_CUDA") and dlib.DLIB_USE_CUDA
    if gpu_available and os.path.exists(mmod_model_path):
        # Use CNN detector (GPU capable)
        detector = dlib.cnn_face_detection_model_v1(mmod_model_path)
    else:
        # fallback HOG (CPU)
        detector = dlib.get_frontal_face_detector()

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Missing predictor: {predictor_path}")
    predictor = dlib.shape_predictor(predictor_path)

    # Small note printed once per worker (goes to worker stdout)
    print(f"[worker init] pid={os.getpid()} gpu_available={gpu_available}")


# ---------------- Video processing function (worker) ----------------
def process_single_video(task):
    """
    task = (speaker, video_filename, video_path, label, save_path)
    Returns: dict with results for logging and progress.
    """
    speaker, video_filename, video_path, label, save_path = task
    start_t = time.time()
    frames_list = []
    skipped_reason = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        skipped_reason = "cannot_open_video"
        return {"speaker": speaker, "video": video_filename, "success": False,
                "reason": skipped_reason, "time": 0.0, "frames": 0, "save_path": save_path}

    tracker = None
    last_bbox = None
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if MAX_FRAMES_PER_VIDEO and frame_idx > MAX_FRAMES_PER_VIDEO:
            break

        # convert to grayscale for predictor & cropping; and RGB for dlib CNN if using it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        do_detect = (frame_idx % DETECT_EVERY_N_FRAMES == 1) or (tracker is None)

        if do_detect:
            # dlib CNN prefers RGB but works with grayscale for detection; we will try RGB for CNN
            try:
                if gpu_available:
                    # convert to rgb for CNN detector (more robust)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    dets = detector(rgb, 1)
                else:
                    dets = detector(gray, 1)  # HOG
            except Exception:
                dets = detector(gray, 1)

            if len(dets) == 0:
                # no face detected in this frame; try to continue by invalidating tracker
                tracker = None
                last_bbox = None
                continue

            # handle cnn face_detection_model_v1 outputs (object with .rect) and HOG rect outputs
            first = dets[0]
            rect = first.rect if hasattr(first, "rect") else first
            # use shape predictor for landmarks (CPU)
            shape = predictor(gray, rect)
            mouth_pts = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]
            mouth_pts_arr = np.array(mouth_pts, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(mouth_pts_arr)

            last_bbox = (x, y, w, h)

            # Start/Reset tracker with the mouth bbox (tracking the mouth area)
            try:
                tracker = cv2.TrackerKCF_create()
            except Exception:
                # CreateTracker in different OpenCV builds:
                try:
                    tracker = cv2.Tracker_create("KCF")
                except Exception:
                    tracker = None

            if tracker is not None:
                # tracker wants (x, y, w, h) in ints and the BGR frame
                try:
                    tracker.init(frame, (x, y, w, h))
                except Exception:
                    tracker = None

            # crop and append
            mouth_img = crop_and_resize_gray(gray, last_bbox)
            if mouth_img is not None:
                frames_list.append(mouth_img)
                processed_frames += 1

        else:
            # use tracker to update bbox
            if tracker is not None:
                ok, tr_bbox = tracker.update(frame)
                if ok:
                    # tr_bbox is (x, y, w, h) float
                    bx, by, bw, bh = [int(v) for v in tr_bbox]
                    last_bbox = (bx, by, bw, bh)
                    mouth_img = crop_and_resize_gray(gray, last_bbox)
                    if mouth_img is not None:
                        frames_list.append(mouth_img)
                        processed_frames += 1
                    else:
                        # tracker gave a bad crop, force re-detection next iteration
                        tracker = None
                        last_bbox = None
                else:
                    # tracker failed; force re-detection
                    tracker = None
                    last_bbox = None
                    continue
            else:
                # no tracker, fallback to detection this frame
                continue

    cap.release()

    if processed_frames < 1:
        skipped_reason = "no_mouth_frames"
        duration = time.time() - start_t
        return {"speaker": speaker, "video": video_filename, "success": False,
                "reason": skipped_reason, "time": duration, "frames": 0, "save_path": save_path}

    # stack frames into array shape (T, 1, H, W) - same convention as earlier code
    video_np = np.stack(frames_list)[:, np.newaxis, :, :]

    # Ensure output dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        np.save(save_path, {"video": video_np, "label": label})
        success = True
    except Exception as e:
        success = False
        skipped_reason = f"save_error:{e}"

    duration = time.time() - start_t
    return {"speaker": speaker, "video": video_filename, "success": success,
            "reason": skipped_reason, "time": duration, "frames": int(processed_frames), "save_path": save_path}


# ---------------- Orchestration (main) ----------------
def main():
    # checks
    if not os.path.exists(GRID_DATASET_PATH):
        raise FileNotFoundError("GRID dataset path not found: " + GRID_DATASET_PATH)
    if not os.path.exists(DLIB_PREDICTOR_PATH):
        raise FileNotFoundError("Missing dlib predictor: " + DLIB_PREDICTOR_PATH)

    detect_gpu_flag = hasattr(dlib, "DLIB_USE_CUDA") and dlib.DLIB_USE_CUDA and os.path.exists(MMOD_MODEL_PATH)
    print(f"[INFO] dlib.DLIB_USE_CUDA = {hasattr(dlib, 'DLIB_USE_CUDA') and dlib.DLIB_USE_CUDA}, using CNN GPU model: {detect_gpu_flag}")
    if detect_gpu_flag:
        try:
            devs = dlib.cuda.get_num_devices()
            print(f"[INFO] dlib reports CUDA device count: {devs}")
        except Exception:
            pass

    speakers = [d for d in os.listdir(GRID_DATASET_PATH) if os.path.isdir(os.path.join(GRID_DATASET_PATH, d))]
    speakers.sort()

    tasks = []
    per_speaker_total = {}
    for spk in speakers:
        spk_path = os.path.join(GRID_DATASET_PATH, spk)
        video_files = [f for f in os.listdir(spk_path) if f.endswith(VIDEO_EXT)]
        video_files.sort()
        per_speaker_total[spk] = len(video_files)

        out_spk_dir = os.path.join(PROCESSED_PATH, spk)
        # Count processed videos for this speaker by checking .npy files on disk
        processed_count = 0
        for vf in video_files:
            save_path = os.path.join(out_spk_dir, vf.replace(VIDEO_EXT, ".npy"))
            if os.path.exists(save_path):
                processed_count += 1

        # If all videos already processed for this speaker, skip entire speaker
        if processed_count >= len(video_files):
            print(f"[SKIP] Speaker {spk} already fully processed ({processed_count}/{len(video_files)}).")
            continue

        # Add tasks only for videos not processed yet
        for vf in video_files:
            save_path = os.path.join(out_spk_dir, vf.replace(VIDEO_EXT, ".npy"))
            if os.path.exists(save_path):
                continue  # skip already processed video
            label = read_label_from_align(spk_path, vf)
            tasks.append((spk, vf, os.path.join(spk_path, vf), label, save_path))

    total_videos = len(tasks)
    print(f"[INFO] Speakers found: {len(speakers)}. Pending videos to process: {total_videos}. Workers: {NUM_WORKERS}")

    if total_videos == 0:
        print("[INFO] Nothing to do - all videos already processed.")
        return

    # open CSV for writing times
    csv_file = open(TIMES_CSV, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    # write header if file is empty
    if os.path.getsize(TIMES_CSV) == 0:
        csv_writer.writerow(["speaker", "video", "success", "reason", "time_sec", "frames", "save_path"])
        csv_file.flush()

    manager = Manager()
    per_speaker_done = manager.dict()
    for spk in speakers:
        per_speaker_done[spk] = 0

    pool = Pool(processes=NUM_WORKERS, initializer=init_worker,
                initargs=(MMOD_MODEL_PATH, DLIB_PREDICTOR_PATH, detect_gpu_flag))
    pbar = tqdm(total=total_videos, desc="Overall videos", unit="video", dynamic_ncols=True)

    try:
        for result in pool.imap_unordered(process_single_video, tasks):
            csv_writer.writerow([result["speaker"], result["video"], result["success"], result["reason"], f"{result['time']:.3f}", result["frames"], result["save_path"]])
            csv_file.flush()

            per_speaker_done[result["speaker"]] = per_speaker_done.get(result["speaker"], 0) + 1

            pbar.update(1)
            pbar.set_postfix({"cur_speaker": result["speaker"], "frames": result["frames"]})
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Received KeyboardInterrupt — terminating workers.")
        pool.terminate()
        pool.join()
    except Exception as e:
        print("[ERROR] Exception during processing:", e)
        pool.terminate()
        pool.join()
    finally:
        pool.close()
        pool.join()
        pbar.close()
        csv_file.close()

    # print per-speaker summary
    print("\nPer-speaker processing summary:")
    for spk in speakers:
        done = int(per_speaker_done.get(spk, 0))
        total = per_speaker_total.get(spk, 0)
        print(f"  {spk}: {done}/{total} videos processed")

    print("\n[✅] All done. Times saved to", TIMES_CSV)


if __name__ == "__main__":
    main()
