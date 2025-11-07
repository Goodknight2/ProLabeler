import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import yt_dlp
import tempfile
import shutil
from collections import deque

# ------------------------------
# YouTube download function
# ------------------------------
def download_youtube_video(url: str, target_res: str = "1080p", progress_callback=None):
    res_int = int(target_res.rstrip("p"))
    temp_dir = tempfile.gettempdir()
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
    output_template = os.path.join(temp_dir, 'yt_video_%(id)s_%(height)sp.%(ext)s')

    def hook(d):
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', '').strip()
            speed = d.get('_speed_str', '').strip()
            eta = d.get('eta', '?')
            text = f"Downloading {percent} at {speed} (ETA: {eta}s)"
            if progress_callback:
                progress_callback(text)
        elif d['status'] == 'finished':
            if progress_callback:
                progress_callback("Download complete, finalizing file...")

    ydl_opts = {
        'format': f'bestvideo[height<={res_int}]+bestaudio/best',
        "ffmpeg_location": ffmpeg_path if os.path.exists(ffmpeg_path) else None,
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'progress_hooks': [hook],
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Failed to download YouTube video: {url}")
    return filename


# ------------------------------
# GPU/CPU detection
# ------------------------------
def get_device(force_cpu=False):
    if force_cpu:
        print("  Forcing CPU mode...")
        return "CPUExecutionProvider"
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        print(" Using GPU: CUDAExecutionProvider")
        return "CUDAExecutionProvider"
    print("  CUDA not available, using CPU.")
    return "CPUExecutionProvider"

# ------------------------------
# Load ONNX model
# ------------------------------
def load_model(model_path, force_cpu=False):
    device = get_device(force_cpu)
    print(f"Loading model {model_path} on {device}")
    
    # Optimize session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # Adjust based on your CPU cores
    sess_options.inter_op_num_threads = 4
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    try:
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=[device, "CPUExecutionProvider"]
        )
    except Exception as e:
        print(f" Failed to load on {device}: {e}")
        print("Retrying on CPU...")
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
    return session

# ------------------------------
# Preprocess frame for YOLOv8
# ------------------------------
def preprocess_frame(frame, target_size=640):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[0:new_h, 0:new_w] = resized

    inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))
    inp = np.expand_dims(inp, axis=0)
    return inp, scale, (new_w, new_h)

# ------------------------------
# Postprocess for 1x5x8400 models
# ------------------------------
def postprocess_1x5x8400(preds, conf_thresh=0.25):
    boxes = []
    preds = preds[0]  # shape 5x8400
    preds = preds.T   # 8400 x 5
    for p in preds:
        conf = p[4]
        if conf >= conf_thresh:
            cx, cy, w, h = p[0], p[1], p[2], p[3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            cls = 0
            boxes.append([x1, y1, x2, y2, conf, cls])
    return boxes

# ------------------------------
# Rescale boxes back to original frame
# ------------------------------
def rescale_boxes(boxes, scale, pad_size, orig_w, orig_h):
    for b in boxes:
        b[0] = b[0] / scale
        b[1] = b[1] / scale
        b[2] = b[2] / scale
        b[3] = b[3] / scale
        b[0] = max(0, min(b[0], orig_w))
        b[1] = max(0, min(b[1], orig_h))
        b[2] = max(0, min(b[2], orig_w))
        b[3] = max(0, min(b[3], orig_h))
    return boxes

# ------------------------------
# Resize image and adjust boxes (crop instead of squish)
# ------------------------------
def resize_and_adjust_boxes(frame, boxes, new_size):
    new_w, new_h = new_size
    orig_h, orig_w = frame.shape[:2]
    orig_aspect = orig_w / orig_h
    new_aspect = new_w / new_h

    if orig_aspect > new_aspect:
        crop_w = int(orig_h * new_aspect)
        x1 = (orig_w - crop_w) // 2
        frame_cropped = frame[:, x1:x1 + crop_w]
        crop_x_offset, crop_y_offset = x1, 0
        scale_x, scale_y = new_w / crop_w, new_h / orig_h
    else:
        crop_h = int(orig_w / new_aspect)
        y1 = (orig_h - crop_h) // 2
        frame_cropped = frame[y1:y1 + crop_h, :]
        crop_x_offset, crop_y_offset = 0, y1
        scale_x, scale_y = new_w / orig_w, new_h / crop_h

    resized = cv2.resize(frame_cropped, (new_w, new_h))
    new_boxes = []
    for b in boxes:
        x1, y1, x2, y2, conf, cls = b
        x1 -= crop_x_offset; x2 -= crop_x_offset
        y1 -= crop_y_offset; y2 -= crop_y_offset
        x1 = max(0, min(x1, frame_cropped.shape[1]))
        x2 = max(0, min(x2, frame_cropped.shape[1]))
        y1 = max(0, min(y1, frame_cropped.shape[0]))
        y2 = max(0, min(y2, frame_cropped.shape[0]))
        x1 *= scale_x; y1 *= scale_y
        x2 *= scale_x; y2 *= scale_y
        if x2 > x1 and y2 > y1:
            new_boxes.append([x1, y1, x2, y2, conf, cls])
    return resized, new_boxes

# ------------------------------
# IoU + box merging
# ------------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def merge_boxes(boxes, merge_iou=0.5):
    merged, used = [], set()
    for i, box in enumerate(boxes):
        if i in used: continue
        x1, y1, x2, y2, conf, cls = box
        for j in range(i + 1, len(boxes)):
            if j in used: continue
            other = boxes[j]
            if cls != other[5]: continue
            if iou(box[:4], other[:4]) > merge_iou:
                x1, y1 = min(x1, other[0]), min(y1, other[1])
                x2, y2 = max(x2, other[2]), max(y2, other[3])
                conf = max(conf, other[4])
                used.add(j)
        merged.append([x1, y1, x2, y2, conf, cls])
    return merged

# ------------------------------
# Perceptual Hash for duplicate detection
# ------------------------------
def compute_phash(frame, hash_size=8):
    """Compute perceptual hash of a frame"""
    # Convert to grayscale and resize to hash_size+1 to allow for DCT
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    
    # Compute horizontal gradient (simple difference hash)
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert to integer hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes"""
    return bin(hash1 ^ hash2).count('1')

def is_similar_to_recent(frame, recent_hashes, similarity_threshold=5):
    """
    Check if frame is similar to any recent frame.
    similarity_threshold: lower = stricter (0 = identical, 64 = completely different)
    """
    current_hash = compute_phash(frame)
    
    for prev_hash in recent_hashes:
        if hamming_distance(current_hash, prev_hash) <= similarity_threshold:
            return True
    
    return False

# ------------------------------
# Save YOLO-format labels
# ------------------------------
def save_labels(boxes, img_w, img_h, save_path):
    lines = []
    for b in boxes:
        x1, y1, x2, y2, conf, cls = [float(v) for v in b]
        x_c = ((x1 + x2) / 2) / img_w
        y_c = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    if lines:
        with open(save_path, "w") as f:
            f.write("\n".join(lines))

# ------------------------------
# Main with cancel + progress + duplicate detection
# ------------------------------
def main(args=None, progress_callback=None, detect_progress_callback=None, cancel_check=None):

    def should_cancel():
        try:
            return bool(cancel_check and cancel_check())
        except Exception:
            return False

    try:
        # Parse CLI if no args passed
        if args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--out-res", type=str, default="off")
            parser.add_argument("--video", type=str)
            parser.add_argument("--youtube", type=str)
            parser.add_argument("--yt-res", type=str, default="1080p")
            parser.add_argument("--models", type=str, nargs="+", required=True)
            parser.add_argument("--out", type=str, required=True)
            parser.add_argument("--conf", type=float, default=0.25)
            parser.add_argument("--iou", type=float, default=0.45)
            parser.add_argument("--merge-iou", type=float, default=0.5)
            parser.add_argument("--frame-step", type=int, default=1)
            parser.add_argument("--cpu", action="store_true")
            parser.add_argument("--similarity-threshold", type=int, default=5, 
                              help="Similarity threshold for duplicate detection (0-64, lower=stricter)")
            parser.add_argument("--history-size", type=int, default=30,
                              help="Number of recent frames to compare against")
            args = parser.parse_args()
        else:
            from types import SimpleNamespace
            args = SimpleNamespace(**args)
            # Set defaults for new parameters if not provided
            if not hasattr(args, 'similarity_threshold'):
                args.similarity_threshold = 5
            if not hasattr(args, 'history_size'):
                args.history_size = 30

        # Source setup
        video_path = None
        if getattr(args, "youtube", None):
            video_path = download_youtube_video(args.youtube, args.yt_res, progress_callback)
            print(f"Downloaded video: {video_path}")
        elif getattr(args, "video", None):
            video_path = args.video
        else:
            raise ValueError("No video or YouTube URL provided.")

        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

        models = [load_model(m, force_cpu=args.cpu) for m in args.models]
        print(f"Loaded {len(models)} models.")
        print(f"Duplicate detection enabled: similarity_threshold={args.similarity_threshold}, history_size={args.history_size}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0
        saved_count = 0
        skipped_duplicates = 0

        # Keep track of recent frame hashes
        recent_hashes = deque(maxlen=args.history_size)

        while True:
            if should_cancel():
                print("Detection canceled by user (before reading).")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if should_cancel():
                print("Detection canceled by user (mid-loop).")
                break

            if frame_idx % args.frame_step != 0:
                frame_idx += 1
                continue

            if frame is None or not hasattr(frame, "shape"):
                frame_idx += 1
                continue

            img_h, img_w = frame.shape[:2]
            ensemble_boxes = []

            # Preprocess once for all models
            inp, scale, pad_size = preprocess_frame(frame)

            # Run models in parallel using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def run_model(model):
                if should_cancel():
                    return []
                input_name = model.get_inputs()[0].name
                pred = model.run(None, {input_name: inp})
                boxes = postprocess_1x5x8400(pred, conf_thresh=args.conf)
                boxes = rescale_boxes(boxes, scale, pad_size, img_w, img_h)
                return boxes
            
            # Use thread pool for parallel inference
            with ThreadPoolExecutor(max_workers=len(models)) as executor:
                futures = [executor.submit(run_model, model) for model in models]
                for future in as_completed(futures):
                    if should_cancel():
                        print("Detection canceled during model inference.")
                        break
                    ensemble_boxes.extend(future.result())

            if should_cancel():
                break

            final_boxes = merge_boxes(ensemble_boxes, merge_iou=args.merge_iou)

            if final_boxes:
                # Check if this frame is similar to recent frames
                if is_similar_to_recent(frame, recent_hashes, args.similarity_threshold):
                    skipped_duplicates += 1
                    frame_idx += 1
                    continue
                
                # Add current frame hash to history
                current_hash = compute_phash(frame)
                recent_hashes.append(current_hash)

                if args.out_res and args.out_res.lower() != "off":
                    try:
                        w, h = map(int, args.out_res.lower().split("x"))
                        frame, final_boxes = resize_and_adjust_boxes(frame, final_boxes, (w, h))
                        img_w, img_h = w, h
                    except Exception:
                        print("Invalid out_res format, expected WIDTHxHEIGHT")

                img_name = f"{frame_idx:06d}.jpg"
                cv2.imwrite(os.path.join(args.out, "images", img_name), frame)
                label_name = f"{frame_idx:06d}.txt"
                save_labels(final_boxes, img_w, img_h, os.path.join(args.out, "labels", label_name))
                saved_count += 1

            frame_idx += 1

            # Update progress bar
            percent = (frame_idx / total_frames) * 100
            remaining = max(0, (total_frames - frame_idx) / fps)
            if detect_progress_callback:
                detect_progress_callback(percent, eta=remaining)

        cap.release()
        print(f"Finished. Saved {saved_count} frames with labels to {args.out}")
        print(f"Skipped {skipped_duplicates} duplicate/similar frames")

    finally:
        # Cleanup temporary YouTube file
        if args and getattr(args, "youtube", None):
            try:
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"Deleted temporary video: {video_path}")
            except Exception as e:
                print(f"Warning: could not delete {video_path}: {e}")


if __name__ == "__main__":
    main()