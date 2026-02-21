# --- recgn_arcnet_superfast_gpu_ct_up_Avid_v10_PYAV.py ---
# --- VERSION: PyAV DECODER + SINGLE-LOADER + FFMPEG THREADING ---

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import sys
import time
import subprocess
import traceback
import ctypes
import argparse
import atexit
import cv2
import gc
import numpy as np
import torch
import threading
from multiprocessing import Process, Queue, Array, set_start_method, Value
from concurrent.futures import ThreadPoolExecutor

# ============ PyAV import ============
try:
    import av
    av.logging.set_level(av.logging.ERROR)
except ImportError:
    print("[!] ERROR: PyAV is not installed. Install it with: pip install av")
    sys.exit(1)

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    print("[!] ERROR: insightface is not installed.")
    sys.exit(1)

# ==========================================
#               SETTINGS
# ==========================================
torch.backends.cudnn.benchmark = True

REFERENCE_FACES_FOLDER = 'samples_jpg'
TARGET_VIDEO_FOLDER = 'in_video'
OUTPUT_FOLDER = 'found_fragments_colored_'
TRT_CACHE_PATH = 'insightface_trt_cache'

MODEL_PACK_NAME = 'buffalo_l'

NUM_GPU_PROCESSES = 5
GPU_WORKER_THREADS = 2

# ===== KEY CHANGE =====
# Use 1-2 sequential decoders per video instead of many seek loaders.
NUM_VIDEO_DECODERS = 2  # 1-2 decoders per video are usually enough
FFMPEG_DECODER_THREADS = 4  # internal FFmpeg decoder threads

# Set True to use PyAV, False to use OpenCV.
USE_PYAV = True

BATCH_SIZE = 8
REC_CHUNK_SIZE = 16

MAX_BUFFER_SLOTS = 1024

DET_SIZE = (1440, 1440)
MAX_PROCESSING_HEIGHT = 1440
MAX_PROCESSING_WIDTH = 2560

CUSTOM_THRESHOLD = 0.49
SEARCH_MODE = 'original_full_frame'
PRE_UPSCALE_FACTOR = 1.0
BOX_PADDING_PERCENTAGE = 0.3

FRAME_INTERVAL = 2
CLIP_DURATION_BEFORE = 1.0
CLIP_DURATION_AFTER = 10.0
MERGE_GAP_TOLERANCE = 12.0

COLOR_PALETTE = ['red', 'lime', 'cyan', 'magenta', 'orange', 'yellow', 'dodgerblue']
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')

QUEUE_FILE = 'recgn_queue.txt'
PROCESSED_QUEUE_FILE = 'recgn_processed_files.txt'
QUEUE_POLL_SECONDS = 20
LOG_FILE_DEFAULT = 'recgn.log'

# ==========================================
#           SYSTEM PATHS
# ==========================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    FFMPEG_DIR = os.path.join(script_dir, "ffmpeg")
    FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")
    if not os.path.exists(FFMPEG_PATH):
        FFMPEG_PATH = "ffmpeg"
        FFPROBE_PATH = "ffprobe"
except:
    FFMPEG_PATH = "ffmpeg"
    FFPROBE_PATH = "ffprobe"


_LOG_STREAM_HANDLE = None
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _close_log_stream():
    global _LOG_STREAM_HANDLE
    if _LOG_STREAM_HANDLE:
        try:
            sys.stdout = _ORIGINAL_STDOUT
            sys.stderr = _ORIGINAL_STDERR
            _LOG_STREAM_HANDLE.flush()
            _LOG_STREAM_HANDLE.close()
        except Exception:
            pass
        _LOG_STREAM_HANDLE = None


def setup_file_logging(log_file):
    global _LOG_STREAM_HANDLE
    if not log_file:
        return None

    log_path = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    _LOG_STREAM_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    session_start = time.strftime("%Y-%m-%d %H:%M:%S")
    _LOG_STREAM_HANDLE.write(f"\n{'=' * 60}\n")
    _LOG_STREAM_HANDLE.write(f"[Session Start] {session_start} | PID={os.getpid()}\n")

    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_STREAM_HANDLE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_STREAM_HANDLE)
    atexit.register(_close_log_stream)
    return log_path


def find_recognition_model(app):
    models = getattr(app, 'models', None)
    if models is None:
        return None
    if isinstance(models, dict):
        iterable = models.values()
    else:
        iterable = models
    for m in iterable:
        if getattr(m, 'taskname', None) == 'recognition':
            return m
    return None


def make_providers():
    trt_options = {
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': TRT_CACHE_PATH,
        'trt_fp16_enable': True,
    }
    return [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider']


def warmup_trt_engine(device_id=0):
    print(f"[*] WARMUP: building TensorRT engine (Device {device_id})...")
    if not os.path.exists(TRT_CACHE_PATH):
        os.makedirs(TRT_CACHE_PATH)

    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        allowed_modules=['detection', 'recognition'],
        providers=make_providers()
    )
    app.prepare(ctx_id=device_id, det_size=DET_SIZE)

    rec_model = find_recognition_model(app)
    if rec_model:
        rec_in = rec_model.session.get_inputs()[0].name
        rec_out = rec_model.session.get_outputs()[0].name
        print("[*] WARMUP: running recognition warmup...")
        fake_blob = np.random.rand(REC_CHUNK_SIZE, 3, 112, 112).astype(np.float32)
        try:
            rec_model.session.run([rec_out], {rec_in: fake_blob})
        except Exception as e:
            print(f"[!] Warmup warning: {e}")

    print("[*] WARMUP: done.")
    del app
    gc.collect()
    time.sleep(1)


def get_color_for_name(name):
    if not name:
        return 'red'
    idx = sum(ord(c) for c in name) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def get_video_duration_and_fps(video_path):
    try:
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path]
        fps_str = subprocess.check_output(cmd).decode().strip()
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 25.0
        else:
            fps = float(fps_str)

        cmd_dur = [FFPROBE_PATH, '-v', 'error', '-show_entries',
                   'format=duration', '-of', 'csv=p=0', video_path]
        dur = float(subprocess.check_output(cmd_dur).decode().strip())
        return dur, fps
    except:
        return None, None

def get_video_info(video_path):
    """Read video metadata using PyAV with OpenCV fallback."""
    
    # Try PyAV first.
    if USE_PYAV:
        try:
            import av
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            w = stream.width
            h = stream.height
            
            # FPS
            if stream.average_rate:
                fps = float(stream.average_rate)
            elif stream.guessed_rate:
                fps = float(stream.guessed_rate)
            else:
                fps = 25.0
            
            # Frame count.
            frames = stream.frames
            if frames == 0 or frames is None:
                # Estimate frame count from duration.
                if stream.duration and stream.time_base:
                    duration_sec = float(stream.duration * stream.time_base)
                    frames = int(duration_sec * fps)
                elif container.duration:
                    duration_sec = container.duration / av.time_base
                    frames = int(duration_sec * fps)
                else:
                    # Final fallback: count frames directly (slow).
                    frames = 0
                    for _ in container.decode(stream):
                        frames += 1
                    container.close()
                    container = av.open(video_path)
            
            container.close()
            
            print(f"[PyAV Info] {video_path}: {w}x{h}, {frames} frames, {fps:.2f} fps")
            return {'frames': frames, 'w': w, 'h': h, 'fps': fps, 'path': video_path}
            
        except Exception as e:
            print(f"[PyAV Info] Failed: {e}, trying OpenCV...")
    
    # Fallback to OpenCV.
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        cap.release()
        
        print(f"[CV2 Info] {video_path}: {w}x{h}, {frames} frames, {fps:.2f} fps")
        return {'frames': frames, 'w': w, 'h': h, 'fps': fps, 'path': video_path}
        
    except Exception as e:
        print(f"[Video Info] Error: {e}")
        return None


def fit_processing_resolution(width, height, max_width=MAX_PROCESSING_WIDTH, max_height=MAX_PROCESSING_HEIGHT):
    """Scale video resolution down to a maximum processing envelope while keeping aspect ratio."""
    if width <= 0 or height <= 0:
        return width, height, 1.0

    scale = min(1.0, min(float(max_width) / float(width), float(max_height) / float(height)))
    scaled_w = max(2, int(round(width * scale)))
    scaled_h = max(2, int(round(height * scale)))

    # Keep even dimensions for stable codec operations.
    if scaled_w % 2 == 1:
        scaled_w -= 1
    if scaled_h % 2 == 1:
        scaled_h -= 1

    return max(2, scaled_w), max(2, scaled_h), scale
        
        
def get_video_info_pyav(video_path):
    """Read video metadata via PyAV."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        frames = stream.frames
        if frames == 0:
            # Estimate frame count when stream metadata is incomplete.
            if stream.duration and stream.time_base:
                duration_sec = float(stream.duration * stream.time_base)
                fps = float(stream.average_rate) if stream.average_rate else 25.0
                frames = int(duration_sec * fps)
            else:
                frames = 0
        
        w = stream.width
        h = stream.height
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        
        container.close()
        return {'frames': frames, 'w': w, 'h': h, 'fps': fps, 'path': video_path}
    except Exception as e:
        print(f"[!] Error reading {video_path}: {e}")
        return None


def save_merged_clip(input_path, output_path, start_time, end_time, detections_in_clip, fps):
    try:
        duration = end_time - start_time
        start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time)) + f'.{int((start_time % 1) * 1000):03d}'

        unique_boxes = []
        last_coords_map = {}
        sorted_dets = sorted(detections_in_clip, key=lambda x: x['time'])
        for det in sorted_dets:
            name = det['name']
            x1, y1, x2, y2 = det['coords']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if name in last_coords_map:
                lx, ly = last_coords_map[name]
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist < 5:
                    continue
            last_coords_map[name] = (cx, cy)
            unique_boxes.append(det)

        filters = []
        for i, det in enumerate(unique_boxes):
            if i > 250:
                break
            x1, y1, x2, y2 = det['coords']
            w, h = x2 - x1, y2 - y1
            color = get_color_for_name(det['name'])
            filters.append(f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color={color}:t=2")

        cmd = [FFMPEG_PATH, '-y', '-ss', start_time_str, '-i', input_path, '-t', str(duration), '-loglevel', 'error']
        if filters:
            vf = ",".join(filters)
            cmd.extend(['-vf', vf, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-c:a', 'copy'])
        else:
            cmd.extend(['-c', 'copy'])
        cmd.append(output_path)

        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        return True, None
    except subprocess.TimeoutExpired:
        return False, "FFMPEG Timeout"
    except Exception as e:
        return False, str(e)


# ==========================================
#         REFERENCE EMBEDDINGS
# ==========================================
def prepare_reference_embeddings(folder_path, device_id):
    if not os.path.exists(folder_path):
        return []
    print(f"[*] Loading reference embeddings (Device: {device_id})...")

    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        allowed_modules=['detection', 'recognition'],
        providers=make_providers()
    )
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    rec_model = find_recognition_model(app)
    if not rec_model:
        sys.exit("[!] Rec model not found")

    rec_in = rec_model.session.get_inputs()[0].name
    rec_out = rec_model.session.get_outputs()[0].name

    mean = float(getattr(rec_model, 'input_mean', 127.5))
    std = float(getattr(rec_model, 'input_std', 127.5))

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
             if f.lower().endswith(('.jpg', '.png'))]
    refs = []
    loaded = 0
    
    for p in files:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            bboxes, kpss = app.det_model.detect(img, max_num=1)
            if not len(bboxes):
                continue
            aimg = face_align.norm_crop(img, landmark=kpss[0])
            blob = cv2.dnn.blobFromImages([aimg], 1.0 / std, (112, 112), (mean, mean, mean), swapRB=True)
            feat = rec_model.session.run([rec_out], {rec_in: blob})[0][0]
            emb = feat / (np.linalg.norm(feat) + 1e-12)
            refs.append({'name': os.path.basename(p), 'embedding': emb})
            loaded += 1
        except:
            pass

    del app
    gc.collect()
    print(f"[*] Reference embeddings loaded: {loaded}")
    return refs


# ==========================================
#            PYAV LOADER
# ==========================================
def frame_loader_pyav(video_path, free_q, filled_q, shared_buf, max_frame_size,
                      shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """
    Sequential PyAV frame loader.
    """
    import av
    av.logging.set_level(av.logging.ERROR)
    
    try:
        h, w, c = shape
        raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)
        interval = settings['frame_interval']
        frame_len = h * w * c

        print(f"[Loader-{loader_id}] Starting: frames {start_f}-{end_f}, interval={interval}")

        # Open input container.
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Decoder threading.
        stream.thread_type = 'AUTO'  # AUTO is safer than FRAME
        stream.thread_count = 0  # auto-detect
        
        # Determine FPS.
        if stream.average_rate:
            fps = float(stream.average_rate)
        elif stream.guessed_rate:
            fps = float(stream.guessed_rate)
        else:
            fps = 25.0
            
        print(f"[Loader-{loader_id}] Video FPS: {fps:.2f}, Resolution: {stream.width}x{stream.height}")

        # Seek close to start frame.
        if start_f > 0:
            target_pts = int(start_f / fps / stream.time_base)
            try:
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                print(f"[Loader-{loader_id}] Seeked to frame ~{start_f}")
            except Exception as e:
                print(f"[Loader-{loader_id}] Seek failed: {e}, starting from beginning")
                container.close()
                container = av.open(video_path)
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'

        batch_indices = []
        batch_f_nums = []
        current_frame = 0
        frames_sent = 0
        
        print(f"[Loader-{loader_id}] Starting decode loop...")

        for packet in container.demux(stream):
            for frame in packet.decode():
                # Compute frame index.
                if frame.pts is not None and stream.time_base:
                    time_sec = float(frame.pts * stream.time_base)
                    current_frame = int(time_sec * fps + 0.5)
                else:
                    current_frame = frame.index if hasattr(frame, 'index') else frames_sent

                # Skip until start frame.
                if current_frame < start_f:
                    continue
                
                # Stop after end frame.
                if current_frame >= end_f:
                    print(f"[Loader-{loader_id}] Reached end_f={end_f}")
                    break

                # Apply frame interval.
                if current_frame % interval != 0:
                    continue

                # Acquire a free slot.
                try:
                    idx = free_q.get(timeout=30)
                except:
                    print(f"[Loader-{loader_id}] Timeout waiting for free slot")
                    break
                    
                if idx is None:
                    print(f"[Loader-{loader_id}] Got None slot, stopping")
                    break

                # Convert frame to BGR.
                try:
                    img = frame.to_ndarray(format='bgr24')
                except Exception as e:
                    print(f"[Loader-{loader_id}] Frame convert error: {e}")
                    free_q.put(idx)
                    continue
                
                # Resize if shape changed.
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))

                # Copy into shared memory.
                offset = idx * max_frame_size
                try:
                    raw_arr[offset: offset + frame_len] = img.ravel()
                except Exception as e:
                    print(f"[Loader-{loader_id}] Memory copy error: {e}")
                    free_q.put(idx)
                    continue
                
                batch_indices.append(idx)
                batch_f_nums.append(current_frame)
                frames_sent += 1

                # Send full batch.
                if len(batch_indices) >= BATCH_SIZE:
                    filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape, orig_shape))
                    with submitted_counter.get_lock():
                        submitted_counter.value += len(batch_indices)
                    batch_indices.clear()
                    batch_f_nums.clear()
                    
            # Exit outer loop once end frame is reached.
            if current_frame >= end_f:
                break

        # Flush remaining frames.
        if batch_indices:
            filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape, orig_shape))
            with submitted_counter.get_lock():
                submitted_counter.value += len(batch_indices)

        container.close()
        print(f"[Loader-{loader_id}] Finished. Sent {frames_sent} frames.")

    except Exception as e:
        print(f"[Loader-{loader_id}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
#          OPENCV LOADER
# ==========================================
def frame_loader_cv2_optimized(video_path, free_q, filled_q, shared_buf, max_frame_size,
                               shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """
    Optimized OpenCV loader used as fallback if PyAV fails.
    """
    try:
        h, w, c = shape
        raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)
        interval = settings['frame_interval']
        frame_len = h * w * c

        print(f"[CV2-Loader-{loader_id}] Starting: frames {start_f}-{end_f}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[CV2-Loader-{loader_id}] Failed to open video!")
            return
            
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Seek close to start frame.
        if start_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        batch_indices = []
        batch_f_nums = []
        current_frame = start_f
        frames_sent = 0

        print(f"[CV2-Loader-{loader_id}] Starting decode loop...")

        while current_frame < end_f:
            ret, frame = cap.read()
            if not ret:
                print(f"[CV2-Loader-{loader_id}] End of video at frame {current_frame}")
                break

            if current_frame % interval == 0:
                try:
                    idx = free_q.get(timeout=30)
                except:
                    print(f"[CV2-Loader-{loader_id}] Timeout waiting for slot")
                    break
                    
                if idx is None:
                    break

                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h))

                offset = idx * max_frame_size
                raw_arr[offset: offset + frame_len] = frame.ravel()
                
                batch_indices.append(idx)
                batch_f_nums.append(current_frame)
                frames_sent += 1

                if len(batch_indices) >= BATCH_SIZE:
                    filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape, orig_shape))
                    with submitted_counter.get_lock():
                        submitted_counter.value += len(batch_indices)
                    batch_indices.clear()
                    batch_f_nums.clear()

            current_frame += 1

        if batch_indices:
            filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape, orig_shape))
            with submitted_counter.get_lock():
                submitted_counter.value += len(batch_indices)

        cap.release()
        print(f"[CV2-Loader-{loader_id}] Finished. Sent {frames_sent} frames.")

    except Exception as e:
        print(f"[CV2-Loader-{loader_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
#          FFMPEG PIPE LOADER
# ==========================================
def frame_loader_ffmpeg_pipe(video_path, free_q, filled_q, shared_buf, max_frame_size,
                             shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """
    FFmpeg subprocess loader with:
    - frame selection filter for FRAME_INTERVAL
    - FFmpeg internal multithreading
    - raw BGR output to stdout

    This can be faster for some codecs due to optimized B-frame skipping.
    """
    try:
        h, w, c = shape
        raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)
        interval = settings['frame_interval']
        frame_len = h * w * c
        fps = settings.get('fps', 25.0)

        # Start time in seconds.
        start_time = start_f / fps
        frames_to_decode = (end_f - start_f) // interval

        # ===== FFmpeg command with frame-selection filter =====
        cmd = [
            FFMPEG_PATH,
            '-threads', str(FFMPEG_DECODER_THREADS),
            '-ss', str(start_time),
            '-i', video_path,
            '-vf', f"select='not(mod(n\\,{interval}))',setpts=N/FRAME_RATE/TB",
            '-frames:v', str(frames_to_decode),
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vsync', 'vfr',
            '-loglevel', 'error',
            'pipe:1'
        ]

        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL,
            bufsize=frame_len * 4  # buffer for several frames
        )

        batch_indices = []
        batch_f_nums = []
        frame_count = start_f

        while frame_count < end_f:
            # Read exactly one frame.
            raw_frame = proc.stdout.read(frame_len)
            if len(raw_frame) != frame_len:
                break

            idx = free_q.get()
            if idx is None:
                proc.terminate()
                break

            offset = idx * max_frame_size
            raw_arr[offset: offset + frame_len] = np.frombuffer(raw_frame, dtype=np.uint8)

            batch_indices.append(idx)
            batch_f_nums.append(frame_count)

            if len(batch_indices) >= BATCH_SIZE:
                filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape, orig_shape))
                with submitted_counter.get_lock():
                    submitted_counter.value += len(batch_indices)
                batch_indices.clear()
                batch_f_nums.clear()

            frame_count += interval

        if batch_indices:
            filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape, orig_shape))
            with submitted_counter.get_lock():
                submitted_counter.value += len(batch_indices)

        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=5)

    except Exception as e:
        print(f"[FFmpeg-Loader-{loader_id} Error] {e}")


# ==========================================
#         DECODER SELECTION
# ==========================================

def frame_loader_optimal(video_path, free_q, filled_q, shared_buf, max_frame_size,
                         shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """Select the preferred loader with automatic fallback."""
    
    if USE_PYAV:
        try:
            import av
            frame_loader_pyav(
                video_path, free_q, filled_q, shared_buf, max_frame_size,
                shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id
            )
        except ImportError:
            print(f"[Loader-{loader_id}] PyAV not available, using OpenCV")
            frame_loader_cv2_optimized(
                video_path, free_q, filled_q, shared_buf, max_frame_size,
                shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id
            )
    else:
        frame_loader_cv2_optimized(
            video_path, free_q, filled_q, shared_buf, max_frame_size,
            shape, orig_shape, start_f, end_f, settings, submitted_counter, loader_id
        )


# ==========================================
#         GPU PREPROCESS
# ==========================================
def gpu_preprocess_scrfd(batch_frames_uint8, target_size,
                         input_mean=127.5, input_std=128.0,
                         swap_rb=True):
    import torch
    import torch.nn.functional as F

    assert batch_frames_uint8.is_cuda
    assert batch_frames_uint8.dtype == torch.uint8

    B, H0, W0, C = batch_frames_uint8.shape
    det_h, det_w = int(target_size[0]), int(target_size[1])

    x = batch_frames_uint8
    if swap_rb:
        x = x[..., [2, 1, 0]]

    x = x.permute(0, 3, 1, 2).contiguous().float()

    scale = min(det_w / float(W0), det_h / float(H0))
    new_w = max(1, int(round(W0 * scale)))
    new_h = max(1, int(round(H0 * scale)))

    resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

    canvas = torch.zeros((B, 3, det_h, det_w), device=resized.device, dtype=torch.float32)
    canvas[:, :, :new_h, :new_w] = resized

    canvas = (canvas - float(input_mean)) / float(input_std)

    prep = {
        "det_scale": float(scale),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "orig_w": int(W0),
        "orig_h": int(H0),
        "det_w": int(det_w),
        "det_h": int(det_h),
    }
    return canvas.contiguous(), prep


# ==========================================
#      THREADED WORKER LOGIC
# ==========================================
def thread_infer_task(det_model, rec_model, ref_matrix, filled_q, free_q, save_q, stats_q,
                      raw_arr, max_frame_size, settings, ref_names):
    # GPU worker loop.
    import warnings
    warnings.filterwarnings("ignore")
    cv2.setNumThreads(0)

    det_sess = det_model.session
    rec_sess = rec_model.session

    det_in_name = det_sess.get_inputs()[0].name
    det_out_names = [o.name for o in det_sess.get_outputs()]

    rec_in_name = rec_sess.get_inputs()[0].name
    rec_out_name = rec_sess.get_outputs()[0].name

    pad_pct = float(settings.get('box_padding_percentage', 0.0))
    threshold = float(settings['threshold'])
    det_shape = settings.get('det_size', (640, 640))

    prob_threshold = float(settings.get('det_prob_threshold', 0.4))
    nms_threshold = float(settings.get('det_nms_threshold', 0.4))
    rec_chunk = int(settings.get('rec_chunk_size', 64))

    det_mean = float(getattr(det_model, 'input_mean', 127.5))
    det_std = float(getattr(det_model, 'input_std', 128.0))
    det_swap = bool(getattr(det_model, 'swapRB', True))

    rec_mean = float(getattr(rec_model, 'input_mean', 127.5))
    rec_std = float(getattr(rec_model, 'input_std', 127.5))

    stride_map = {
        8: {"score": 0, "bbox": 3, "kps": 6},
        16: {"score": 1, "bbox": 4, "kps": 7},
        32: {"score": 2, "bbox": 5, "kps": 8},
    }

    input_height, input_width = int(det_shape[0]), int(det_shape[1])
    _num_anchors = 2

    center_cache = {}
    for stride in (8, 16, 32):
        feat_h = input_height // stride
        feat_w = input_width // stride
        xs = np.arange(feat_w, dtype=np.float32)
        ys = np.arange(feat_h, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        centers = centers * stride
        centers = np.repeat(centers, _num_anchors, axis=0)
        center_cache[stride] = centers

    full_mem_view = memoryview(raw_arr)
    stream = torch.cuda.Stream()
    device_id = torch.cuda.current_device()

    # =========================================================
    # Pinned-memory staging buffers for faster async H2D copies.
    # =========================================================
    pinned_host = None        # CPU torch.Tensor with pin_memory=True
    pinned_host_np = None     # NumPy view over pinned_host
    pinned_shape = None       # (h, w, c)
    pinned_capacity = int(BATCH_SIZE)  # max batch size from loader

    while True:
        task = filled_q.get()
        if task is None:
            filled_q.put(None)
            break

        t0 = time.perf_counter()
        if len(task) >= 5:
            video_path, f_nums, indices, shape, orig_shape = task
        else:
            video_path, f_nums, indices, shape = task
            orig_shape = shape
        h, w, c = shape
        orig_h, orig_w, _ = orig_shape
        scale_x = float(orig_w) / float(w) if w > 0 else 1.0
        scale_y = float(orig_h) / float(h) if h > 0 else 1.0
        batch_sz = len(indices)
        frame_len = h * w * c

        all_crops = []
        meta = []

        with torch.cuda.stream(stream):
            # =========================================================
            # Fast fetch into pinned memory + async H2D transfer.
            # =========================================================
            t_f = time.perf_counter()

            # Recreate pinned buffer when resolution changes.
            if (pinned_host is None) or (pinned_shape != (h, w, c)):
                pinned_host = torch.empty(
                    (pinned_capacity, h, w, c),
                    dtype=torch.uint8,
                    device='cpu',
                    pin_memory=True
                )
                pinned_host_np = pinned_host.numpy()
                pinned_shape = (h, w, c)

            # Copy shared-memory frames into pinned CPU memory.
            # This enables faster non-blocking H2D transfer.
            for k, idx in enumerate(indices):
                off = idx * max_frame_size
                src = np.frombuffer(
                    full_mem_view[off: off + frame_len],
                    dtype=np.uint8
                ).reshape(h, w, c)
                np.copyto(pinned_host_np[k], src)

            # NumPy view for face_align / save operations.
            batch_np = pinned_host_np[:batch_sz]

            # Async H2D transfer (effective with pinned memory).
            gpu_frames = pinned_host[:batch_sz].to('cuda', non_blocking=True)

            t_fetch = time.perf_counter() - t_f
            # =========================================================

            t_i = time.perf_counter()
            det_input_batch, prep = gpu_preprocess_scrfd(
                gpu_frames, det_shape, input_mean=det_mean, input_std=det_std, swap_rb=det_swap
            )
            stream.synchronize()

            det_scale = float(prep["det_scale"])
            valid_w = int(prep["new_w"])
            valid_h = int(prep["new_h"])

            # Detection per frame (ONNX detector uses batch=1 here).
            for i in range(batch_sz):
                det_binding = det_sess.io_binding()
                det_in = det_input_batch[i:i + 1].contiguous()

                det_binding.bind_input(
                    name=det_in_name,
                    device_type='cuda',
                    device_id=device_id,
                    element_type=np.float32,
                    shape=tuple(det_in.shape),
                    buffer_ptr=det_in.data_ptr()
                )
                for out_name in det_out_names:
                    det_binding.bind_output(out_name, 'cpu')

                det_sess.run_with_iobinding(det_binding)
                det_outs = det_binding.copy_outputs_to_cpu()

                frame_preds = []
                frame_kpss = []

                for stride in (8, 16, 32):
                    feat_h = input_height // stride
                    feat_w = input_width // stride
                    N = feat_h * feat_w * _num_anchors
                    centers = center_cache[stride]

                    scores = det_outs[stride_map[stride]["score"]].reshape(-1).astype(np.float32)
                    bbox_preds = det_outs[stride_map[stride]["bbox"]].reshape(-1, 4).astype(np.float32)
                    kps_preds = det_outs[stride_map[stride]["kps"]].reshape(-1, 10).astype(np.float32)

                    idx_keep = np.where(scores >= prob_threshold)[0]
                    if idx_keep.size == 0:
                        continue
                    if idx_keep.max() >= centers.shape[0]:
                        continue

                    ac = centers[idx_keep]
                    sc = scores[idx_keep]
                    bb = bbox_preds[idx_keep]
                    kp = kps_preds[idx_keep]

                    m = (ac[:, 0] < valid_w) & (ac[:, 1] < valid_h)
                    if not np.any(m):
                        continue
                    ac, sc, bb, kp = ac[m], sc[m], bb[m], kp[m]

                    x1 = (ac[:, 0] - bb[:, 0] * stride) / det_scale
                    y1 = (ac[:, 1] - bb[:, 1] * stride) / det_scale
                    x2 = (ac[:, 0] + bb[:, 2] * stride) / det_scale
                    y2 = (ac[:, 1] + bb[:, 3] * stride) / det_scale
                    boxes = np.stack([x1, y1, x2, y2, sc], axis=-1)

                    kpss = np.zeros((kp.shape[0], 10), dtype=np.float32)
                    for kk in range(5):
                        kpss[:, kk * 2] = (ac[:, 0] + kp[:, kk * 2] * stride) / det_scale
                        kpss[:, kk * 2 + 1] = (ac[:, 1] + kp[:, kk * 2 + 1] * stride) / det_scale

                    frame_preds.append(boxes)
                    frame_kpss.append(kpss)

                if not frame_preds:
                    continue

                frame_preds = np.concatenate(frame_preds, axis=0)
                frame_kpss = np.concatenate(frame_kpss, axis=0)

                nms_boxes = frame_preds[:, :4].copy()
                nms_boxes[:, 2] -= nms_boxes[:, 0]
                nms_boxes[:, 3] -= nms_boxes[:, 1]
                keep = cv2.dnn.NMSBoxes(
                    nms_boxes.tolist(), frame_preds[:, 4].tolist(),
                    prob_threshold, nms_threshold
                )
                if len(keep) == 0:
                    continue
                keep = np.array(keep).flatten()
                det_boxes = frame_preds[keep]
                det_kpss = frame_kpss[keep]

                np.clip(det_boxes[:, 0], 0, w - 1, out=det_boxes[:, 0])
                np.clip(det_boxes[:, 1], 0, h - 1, out=det_boxes[:, 1])
                np.clip(det_boxes[:, 2], 0, w - 1, out=det_boxes[:, 2])
                np.clip(det_boxes[:, 3], 0, h - 1, out=det_boxes[:, 3])

                orig_img = batch_np[i]
                for box, kps in zip(det_boxes, det_kpss):
                    kps = kps.reshape(5, 2)
                    try:
                        aimg = face_align.norm_crop(orig_img, landmark=kps)
                        all_crops.append(aimg)

                        b = box.astype(np.int32)
                        x1i, y1i, x2i, y2i = b[:4]
                        bw, bh = x2i - x1i, y2i - y1i
                        px, py = int(bw * pad_pct), int(bh * pad_pct)
                        nx1 = max(0, x1i - px)
                        ny1 = max(0, y1i - py)
                        nx2 = min(w, x2i + px)
                        ny2 = min(h, y2i + py)
                        src_nx1 = int(round(nx1 * scale_x))
                        src_ny1 = int(round(ny1 * scale_y))
                        src_nx2 = int(round(nx2 * scale_x))
                        src_ny2 = int(round(ny2 * scale_y))

                        src_nx1 = max(0, min(orig_w - 1, src_nx1))
                        src_ny1 = max(0, min(orig_h - 1, src_ny1))
                        src_nx2 = max(0, min(orig_w, src_nx2))
                        src_ny2 = max(0, min(orig_h, src_ny2))

                        meta.append(
                            (
                                i,
                                (nx1, ny1, nx2, ny2),
                                ((nx1 + nx2) >> 1, (ny1 + ny2) >> 1),
                                (src_nx1, src_ny1, src_nx2, src_ny2),
                                ((src_nx1 + src_nx2) >> 1, (src_ny1 + src_ny2) >> 1),
                            )
                        )
                    except:
                        pass

            faces_found = 0
            t_match_sum = 0.0
            res_map = [[] for _ in range(batch_sz)]

            if all_crops:
                t_m = time.perf_counter()
                for start in range(0, len(all_crops), rec_chunk):
                    chunk_crops = all_crops[start:start + rec_chunk]
                    chunk_meta = meta[start:start + rec_chunk]

                    blob = cv2.dnn.blobFromImages(
                        chunk_crops, 1.0 / rec_std, (112, 112),
                        (rec_mean, rec_mean, rec_mean), swapRB=True
                    )
                    feats = rec_sess.run([rec_out_name], {rec_in_name: blob})[0]
                    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

                    sims = np.dot(feats, ref_matrix.T)
                    best_indices = np.argmax(sims, axis=1)
                    best_scores = sims[np.arange(sims.shape[0]), best_indices]

                    for k, score in enumerate(best_scores):
                        dist = 1.0 - float(score)
                        if dist <= threshold:
                            fi, co, ce, co_src, ce_src = chunk_meta[k]

                            dup = False
                            for ex in res_map[fi]:
                                if abs(ce_src[0] - ex['center_src'][0]) < 20 and abs(ce_src[1] - ex['center_src'][1]) < 20:
                                    dup = True
                                    break
                            if not dup:
                                faces_found += 1
                                res_map[fi].append({
                                    'ref_idx': int(best_indices[k]),
                                    'dist': dist,
                                    'coords': co,
                                    'coords_src': co_src,
                                    'center': ce,
                                    'center_src': ce_src,
                                    'tag': ""
                                })

                t_match_sum = time.perf_counter() - t_m

            t_infer = time.perf_counter() - t_i

            if faces_found > 0:
                for i in range(batch_sz):
                    if res_map[i]:
                        payload = [{
                            'name': ref_names[x['ref_idx']],
                            'dist': x['dist'],
                            'coords': x['coords'],
                            'coords_src': x['coords_src'],
                            'tag': ""
                        } for x in res_map[i]]
                        save_q.put(('data', video_path, f_nums[i], batch_np[i].copy(), payload))

        for idx in indices:
            free_q.put(idx)

        stats_q.put(('metrics', (t_fetch, 0.0, t_infer, t_match_sum, time.perf_counter() - t0, faces_found), batch_sz))
        
        
        
def gpu_manager_process(device_str, filled_q, free_q, save_q, stats_q,
                        shared_buf, max_frame_size, ref_embs, settings, num_threads, proc_rank):
    try:
        gpu_id = int(device_str.split(':')[-1])
    except:
        gpu_id = 0

    cv2.setNumThreads(0)
    torch.cuda.set_device(gpu_id)

    print(f"[GPU-MANAGER-{proc_rank}] Init on {device_str} with {num_threads} THREADS...")

    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        allowed_modules=['detection', 'recognition'],
        providers=make_providers()
    )
    app.prepare(ctx_id=gpu_id, det_size=DET_SIZE)

    det_model = app.det_model
    rec_model = find_recognition_model(app)

    ref_matrix = np.array([r['embedding'] for r in ref_embs], dtype=np.float32)
    ref_names = [r['name'] for r in ref_embs]
    raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)

    print(f"[GPU-MANAGER-{proc_rank}] Ready.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(num_threads):
            futures.append(
                executor.submit(
                    thread_infer_task,
                    det_model, rec_model, ref_matrix,
                    filled_q, free_q, save_q, stats_q,
                    raw_arr, max_frame_size, settings, ref_names
                )
            )
        for f in futures:
            f.result()

    print(f"[GPU-MANAGER-{proc_rank}] Done.")
    del app


# --- SAVER ---
def result_saver(save_q, out_dir, settings):
    print(f"[Saver] Started.")
    os.makedirs(out_dir, exist_ok=True)

    video_meta_cache = {}
    detection_buffer = {}

    while True:
        try:
            task = save_q.get()
            if task is None:
                break

            msg_type = task[0]

            if msg_type == 'data':
                _, video_path, f, img, faces_payload = task

                if video_path not in video_meta_cache:
                    dur, fps = get_video_duration_and_fps(video_path)
                    video_meta_cache[video_path] = {'fps': fps or 25.0, 'dur': dur or 99999}
                    detection_buffer[video_path] = []
                    os.makedirs(os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0], 'img_jpg'),
                                exist_ok=True)

                sec = f / video_meta_cache[video_path]['fps']
                for face in faces_payload:
                    clean_name = os.path.splitext(face['name'])[0]
                    source_coords = face.get('coords_src', face['coords'])
                    detection_buffer[video_path].append(
                        {'time': sec, 'name': clean_name, 'dist': face['dist'], 'coords': source_coords}
                    )

                    try:
                        img_copy = img.copy()
                        x1, y1, x2, y2 = face['coords']
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{clean_name} {face['dist']:.2f}"
                        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        fname = f"f{f}_{clean_name}_{int(time.time() * 100) % 1000}.jpg"
                        save_path = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0], 'img_jpg', fname)
                        cv2.imwrite(save_path, img_copy)
                    except Exception as e:
                        print(f"[Saver Error] ImgSave: {e}")

            elif msg_type == 'end_video':
                _, video_path = task
                if video_path in detection_buffer and detection_buffer[video_path]:
                    dets = sorted(detection_buffer[video_path], key=lambda x: x['time'])
                    meta = video_meta_cache.get(video_path)

                    merged = []
                    if dets:
                        cur_d = [dets[0]]
                        cur_s = max(0, dets[0]['time'] - CLIP_DURATION_BEFORE)
                        cur_e = min(meta['dur'], dets[0]['time'] + CLIP_DURATION_AFTER)

                        for ii in range(1, len(dets)):
                            d = dets[ii]
                            ds = max(0, d['time'] - CLIP_DURATION_BEFORE)
                            de = min(meta['dur'], d['time'] + CLIP_DURATION_AFTER)

                            if ds - cur_e < MERGE_GAP_TOLERANCE:
                                cur_e = max(cur_e, de)
                                cur_d.append(d)
                            else:
                                merged.append((cur_s, cur_e, cur_d))
                                cur_s, cur_e, cur_d = ds, de, [d]

                        merged.append((cur_s, cur_e, cur_d))

                    sub = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0])
                    os.makedirs(sub, exist_ok=True)

                    for s, e, d_list in merged:
                        min_dist = min(x['dist'] for x in d_list)
                        names = sorted(list(set(x['name'] for x in d_list)))
                        names_str = '_'.join(names)[:50]
                        vname = f"{int(s)}s_{names_str}_d{min_dist:.2f}.mp4"
                        save_merged_clip(video_path, os.path.join(sub, vname), s, e, d_list, meta['fps'])

                if video_path in detection_buffer:
                    del detection_buffer[video_path]

        except Exception as e:
            print(f"[!] Saver Error: {e}")
            traceback.print_exc()


def _ensure_text_file(path):
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8'):
            pass


def _read_non_empty_lines(path):
    _ensure_text_file(path)
    lines = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip().lstrip('\ufeff')
                if line:
                    lines.append(line)
    except Exception as e:
        print(f"[Queue] Cannot read {path}: {e}")
    return lines


def _append_unique_lines(path, values):
    _ensure_text_file(path)
    existing = set(_read_non_empty_lines(path))
    to_append = []
    for value in values:
        line = str(value).strip()
        if not line or line in existing:
            continue
        existing.add(line)
        to_append.append(line)

    if not to_append:
        return 0

    with open(path, 'a', encoding='utf-8') as f:
        for line in to_append:
            f.write(line + '\n')
    return len(to_append)


def _normalize_video_file_list(input_files):
    files = []
    seen = set()
    for raw_path in input_files:
        if raw_path is None:
            continue
        path = os.path.abspath(str(raw_path).strip().strip('"'))
        if not path or path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            print(f"[Queue] Skip missing file: {path}")
            continue
        if not path.lower().endswith(VIDEO_EXTENSIONS):
            print(f"[Queue] Skip unsupported extension: {path}")
            continue
        files.append(path)
    return files


def _collect_pending_queue_files(queue_file, processed_file, max_items=0):
    queued = _read_non_empty_lines(queue_file)
    processed = set(_read_non_empty_lines(processed_file))
    pending = []
    seen = set()

    for item in queued:
        path = os.path.abspath(item.strip().strip('"'))
        if not path or path in seen or path in processed:
            continue
        seen.add(path)

        if not os.path.isfile(path):
            print(f"[Queue] Skip missing queued file: {path}")
            continue
        if not path.lower().endswith(VIDEO_EXTENSIONS):
            print(f"[Queue] Skip unsupported queued extension: {path}")
            continue

        pending.append(path)
        if max_items and len(pending) >= max_items:
            break

    return pending


def _build_video_infos(files):
    video_infos = []
    for path in files:
        info = get_video_info(path)
        if not info:
            continue
        if info['frames'] == 0:
            print(f"[!] Skip {path}: no frames detected")
            continue

        process_cnt = (info['frames'] + FRAME_INTERVAL - 1) // FRAME_INTERVAL
        proc_w, proc_h, resize_scale = fit_processing_resolution(info['w'], info['h'])
        info['proc_frames'] = process_cnt
        info['proc_w'] = proc_w
        info['proc_h'] = proc_h
        info['resize_scale'] = resize_scale
        video_infos.append(info)
    return video_infos


def _allocate_shared_buffer(max_frame_size, requested_slots):
    slots = max(32, int(requested_slots))
    while slots >= 32:
        try:
            shared = Array(ctypes.c_uint8, slots * max_frame_size, lock=False)
            return shared, slots
        except Exception as e:
            print(f"[!] Shared memory allocation failed for {slots} slots: {e}")
            if slots == 32:
                break
            slots = max(32, slots // 2)
    return None, 0


def _create_runtime(refs):
    settings = {
        'frame_interval': FRAME_INTERVAL,
        'clip_before': CLIP_DURATION_BEFORE,
        'clip_after': CLIP_DURATION_AFTER,
        'threshold': CUSTOM_THRESHOLD,
        'search_mode': SEARCH_MODE,
        'pre_upscale_factor': PRE_UPSCALE_FACTOR,
        'box_padding_percentage': BOX_PADDING_PERCENTAGE,
        'rec_chunk_size': REC_CHUNK_SIZE,
        'det_prob_threshold': 0.25,
        'det_nms_threshold': 0.4,
        'det_size': DET_SIZE,
    }

    max_h = MAX_PROCESSING_HEIGHT
    max_w = MAX_PROCESSING_WIDTH
    max_frame_size = max_h * max_w * 3

    print(f"[*] Max processing envelope: {max_w}x{max_h} (<= 1440p)")
    print(f"[*] Requested shared memory slots: {MAX_BUFFER_SLOTS}")
    print(f"[*] Per-slot size: {max_frame_size / 1024 / 1024:.1f} MB")

    shared, num_buf = _allocate_shared_buffer(max_frame_size, MAX_BUFFER_SLOTS)
    if shared is None or num_buf <= 0:
        print("[!] Unable to allocate shared memory.")
        return None

    print(f"[*] Shared memory slots in use: {num_buf}")

    free_q = Queue()
    filled_q = Queue()
    save_q = Queue()
    stats_q = Queue()

    for idx in range(num_buf):
        free_q.put(idx)

    saver = Process(target=result_saver, args=(save_q, OUTPUT_FOLDER, settings))
    saver.start()

    print(f"[*] Starting {NUM_GPU_PROCESSES} GPU processes x {GPU_WORKER_THREADS} threads...")
    gpu_processes = []
    for rank in range(NUM_GPU_PROCESSES):
        process = Process(
            target=gpu_manager_process,
            args=(f'cuda:0', filled_q, free_q, save_q, stats_q,
                  shared, max_frame_size, refs, settings, GPU_WORKER_THREADS, rank)
        )
        process.start()
        gpu_processes.append(process)

    time.sleep(3)
    print("[*] GPU processes started, checking status...")
    for idx, process in enumerate(gpu_processes):
        print(f"    GPU-{idx}: alive={process.is_alive()}")

    return {
        'shared': shared,
        'max_frame_size': max_frame_size,
        'free_q': free_q,
        'filled_q': filled_q,
        'save_q': save_q,
        'stats_q': stats_q,
        'settings': settings,
        'saver': saver,
        'gpu_processes': gpu_processes,
    }


def _prepare_runtime():
    if not torch.cuda.is_available():
        print("[!] CUDA is not available.")
        return None

    os.makedirs(TRT_CACHE_PATH, exist_ok=True)
    warmup_trt_engine(0)

    refs = prepare_reference_embeddings(REFERENCE_FACES_FOLDER, 0)
    if not refs:
        print("[!] No reference embeddings were loaded.")
        return None

    return _create_runtime(refs)


def _stop_runtime(runtime):
    if not runtime:
        return

    print("\n[*] Stopping GPU processes...")
    filled_q = runtime['filled_q']
    save_q = runtime['save_q']
    gpu_processes = runtime['gpu_processes']
    saver = runtime['saver']

    for _ in range(NUM_GPU_PROCESSES):
        filled_q.put(None)
    for process in gpu_processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()

    save_q.put(None)
    saver.join(timeout=10)
    if saver.is_alive():
        saver.terminate()


def _drain_metrics_queue(stats_q):
    processed_count = 0
    faces_count = 0
    while True:
        try:
            msg = stats_q.get_nowait()
        except Exception:
            break
        if isinstance(msg, tuple) and msg and msg[0] == 'metrics':
            processed_count += int(msg[2])
            faces_count += int(msg[1][5])
    return processed_count, faces_count


def _process_videos_with_runtime(video_infos, runtime):
    shared = runtime['shared']
    max_frame_size = runtime['max_frame_size']
    free_q = runtime['free_q']
    filled_q = runtime['filled_q']
    save_q = runtime['save_q']
    stats_q = runtime['stats_q']
    settings = runtime['settings']

    processed_videos = []

    for info in video_infos:
        vid_path = info['path']
        total_frames_to_proc = info['proc_frames']
        proc_w = int(info['proc_w'])
        proc_h = int(info['proc_h'])
        resize_scale = float(info.get('resize_scale', 1.0))

        print(f"\n{'=' * 60}")
        print(f">> VIDEO: {os.path.basename(vid_path)}")
        print(f">> Total: {info['frames']} frames | Process: {total_frames_to_proc}")
        print(f">> Source resolution: {info['w']}x{info['h']} @ {info['fps']:.2f} fps")
        if resize_scale < 0.999:
            print(f">> Processing resolution: {proc_w}x{proc_h} (scale={resize_scale:.4f})")
        else:
            print(f">> Processing resolution: {proc_w}x{proc_h}")
        print(f"{'=' * 60}")

        # Drop stale metrics from previous video before starting a new one.
        _drain_metrics_queue(stats_q)

        v_start_t = time.time()
        submitted = Value('i', 0)
        loaders = []

        print(f"[*] Starting {NUM_VIDEO_DECODERS} decoder(s)...")
        proc_shape = (proc_h, proc_w, 3)
        orig_shape = (int(info['h']), int(info['w']), 3)

        if NUM_VIDEO_DECODERS == 1:
            loader = Process(
                target=frame_loader_optimal,
                args=(vid_path, free_q, filled_q, shared, max_frame_size,
                      proc_shape, orig_shape, 0, info['frames'],
                      {**settings, 'fps': info['fps']}, submitted, 0)
            )
            loaders.append(loader)
        else:
            chunk = info['frames'] // NUM_VIDEO_DECODERS
            for idx in range(NUM_VIDEO_DECODERS):
                start_frame = idx * chunk
                end_frame = (idx + 1) * chunk if idx != NUM_VIDEO_DECODERS - 1 else info['frames']
                loader = Process(
                    target=frame_loader_optimal,
                    args=(vid_path, free_q, filled_q, shared, max_frame_size,
                          proc_shape, orig_shape, start_frame, end_frame,
                          {**settings, 'fps': info['fps']}, submitted, idx)
                )
                loaders.append(loader)

        for loader in loaders:
            loader.start()

        time.sleep(0.5)
        print(f"[*] Loaders started: {[loader.is_alive() for loader in loaders]}")

        processed_count = 0
        total_faces_video = 0
        last_log_time = time.time()
        last_submitted = 0
        stall_count = 0
        submitted_val = 0
        video_completed = False

        while True:
            count_add, faces_add = _drain_metrics_queue(stats_q)
            processed_count += count_add
            total_faces_video += faces_add

            with submitted.get_lock():
                submitted_val = submitted.value

            active_loaders = any(loader.is_alive() for loader in loaders)

            now = time.time()
            if now - last_log_time > 1.0:
                elapsed = now - v_start_t
                fps = processed_count / elapsed if elapsed > 0.1 else 0.0
                pct = (processed_count / total_frames_to_proc) * 100 if total_frames_to_proc > 0 else 0

                try:
                    q_in_sz = filled_q.qsize()
                    q_out_sz = free_q.qsize()
                except Exception:
                    q_in_sz = '?'
                    q_out_sz = '?'

                loader_status = 'ALIVE' if active_loaders else 'DONE'
                stat_line = (f"\r[{loader_status}] {pct:5.1f}% | {processed_count}/{total_frames_to_proc} | "
                             f"Submitted: {submitted_val} | FPS: {fps:5.1f} | Faces: {total_faces_video} | "
                             f"Q:{q_in_sz}/{q_out_sz}   ")
                sys.stdout.write(stat_line)
                sys.stdout.flush()
                last_log_time = now

                if submitted_val == last_submitted and active_loaders:
                    stall_count += 1
                    if stall_count > 10:
                        print(f"\n[!] WARNING: Loader may be stalled. Submitted={submitted_val}")
                else:
                    stall_count = 0
                last_submitted = submitted_val

            all_loaders_done = not active_loaders
            all_processed = processed_count >= submitted_val

            if all_loaders_done and all_processed and submitted_val > 0:
                video_completed = True
                break

            if all_loaders_done and submitted_val == 0:
                print("\n[!] ERROR: Loaders finished but submitted 0 frames!")
                video_completed = False
                break

            time.sleep(0.05)

        print("")

        for loader in loaders:
            if loader.is_alive():
                loader.terminate()
            loader.join(timeout=5)

        save_q.put(('end_video', vid_path))

        v_dur = time.time() - v_start_t
        final_fps = processed_count / v_dur if v_dur > 0 else 0

        print(f"OK: {os.path.basename(vid_path)}")
        print(f"  Time: {v_dur:.2f}s | FPS: {final_fps:.1f} | Faces: {total_faces_video}")
        if video_completed:
            processed_videos.append(os.path.abspath(vid_path))
        else:
            print(f"[!] Not marking as processed: {os.path.basename(vid_path)}")

    return processed_videos


def run_queue_worker(queue_file=QUEUE_FILE, processed_file=PROCESSED_QUEUE_FILE,
                     poll_seconds=QUEUE_POLL_SECONDS, once=False, max_items=0):
    queue_file = os.path.abspath(queue_file)
    processed_file = os.path.abspath(processed_file)
    _ensure_text_file(queue_file)
    _ensure_text_file(processed_file)

    print("=" * 60)
    print("  QUEUE MODE - recgn.py")
    print("=" * 60)
    print(f"[Queue] queue_file: {queue_file}")
    print(f"[Queue] processed_file: {processed_file}")
    print(f"[Queue] poll_seconds: {poll_seconds}")

    runtime = None
    if not once:
        runtime = _prepare_runtime()
        if runtime is None:
            return 1

    try:
        while True:
            pending = _collect_pending_queue_files(queue_file, processed_file, max_items=max_items)
            if not pending:
                if once:
                    print("[Queue] No pending files.")
                    return 0
                time.sleep(max(1.0, float(poll_seconds)))
                continue

            print(f"[Queue] Pending videos: {len(pending)}")
            try:
                if once:
                    processed_now = run(input_files=pending)
                else:
                    processed_now = run(input_files=pending, runtime=runtime)
            except Exception as e:
                print(f"[Queue] Processing cycle failed: {e}")
                traceback.print_exc()
                processed_now = []

            if processed_now:
                written = _append_unique_lines(processed_file, processed_now)
                print(f"[Queue] Marked processed: {written}")
            else:
                print("[Queue] Nothing marked as processed in this cycle.")

            if once:
                return 0
    finally:
        if runtime is not None:
            _stop_runtime(runtime)


# --- MAIN ---
def run(input_files=None, runtime=None):
    print("=" * 60)
    print("  FACE RECOGNITION - FIXED DECODER")
    print("=" * 60)
    t_global_start = time.time()

    if input_files is None:
        files = [os.path.join(TARGET_VIDEO_FOLDER, f) for f in os.listdir(TARGET_VIDEO_FOLDER)
                 if f.lower().endswith(VIDEO_EXTENSIONS)]
        files.sort()
    else:
        files = _normalize_video_file_list(input_files)
    if not files:
        print("[!] No input videos found.")
        return []

    video_infos = _build_video_infos(files)
    if not video_infos:
        print("[!] No valid videos found")
        return []

    owns_runtime = runtime is None
    active_runtime = runtime

    if owns_runtime:
        active_runtime = _prepare_runtime()
        if active_runtime is None:
            return []

    try:
        processed_videos = _process_videos_with_runtime(video_infos, active_runtime)
    finally:
        if owns_runtime:
            _stop_runtime(active_runtime)

    total_time = time.time() - t_global_start
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE. Total: {total_time:.2f}s")
    print(f"{'=' * 60}")
    return processed_videos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face recognition pipeline with optional file queue mode."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help="Optional explicit list of videos for one-time processing.",
    )
    parser.add_argument(
        "--queue-mode",
        action="store_true",
        help="Run forever in queue-consumer mode.",
    )
    parser.add_argument(
        "--queue-once",
        action="store_true",
        help="Process queue one time and exit.",
    )
    parser.add_argument(
        "--queue-file",
        default=QUEUE_FILE,
        help=f"Queue file path (default: {QUEUE_FILE}).",
    )
    parser.add_argument(
        "--processed-file",
        default=PROCESSED_QUEUE_FILE,
        help=f"Processed-file path (default: {PROCESSED_QUEUE_FILE}).",
    )
    parser.add_argument(
        "--queue-poll-seconds",
        type=float,
        default=QUEUE_POLL_SECONDS,
        help=f"Polling interval in seconds (default: {QUEUE_POLL_SECONDS}).",
    )
    parser.add_argument(
        "--max-queue-items",
        type=int,
        default=0,
        help="Max queue items per cycle. 0 means no limit.",
    )
    parser.add_argument(
        "--log-file",
        default=LOG_FILE_DEFAULT,
        help=f"Log file path for stdout/stderr tee (default: {LOG_FILE_DEFAULT}).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    active_log_file = setup_file_logging(args.log_file)
    if active_log_file:
        print(f"[*] Logging to: {active_log_file}")
    set_start_method('spawn', force=True)

    if args.queue_mode or args.queue_once:
        run_queue_worker(
            queue_file=args.queue_file,
            processed_file=args.processed_file,
            poll_seconds=args.queue_poll_seconds,
            once=args.queue_once,
            max_items=max(0, args.max_queue_items),
        )
    elif args.videos:
        run(input_files=args.videos)
    else:
        run()
