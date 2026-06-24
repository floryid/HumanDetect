import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _require(module_name: str, pip_hint: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Modul '{module_name}' tidak tersedia. Install dulu: {pip_hint}"
        ) from exc


_require("numpy", "pip install numpy")

import numpy as np  # noqa: E402

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _try_import_pil_image():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None



def _try_import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception:
        return None


def _try_import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore

        return ort
    except Exception:
        return None


def _try_import_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter
    except Exception:
        try:
            import tensorflow as tf  # type: ignore

            return tf.lite.Interpreter
        except Exception:
            return None


def _resize_bgr(img_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    if cv2 is not None:
        return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    Image = _try_import_pil_image()
    if Image is None:
        raise RuntimeError("Butuh pillow jika opencv tidak tersedia. Install: pip install pillow")
    rgb = img_bgr[..., ::-1]
    im = Image.fromarray(rgb)
    im = im.resize((w, h))
    rgb2 = np.array(im)
    if rgb2.ndim == 2:
        rgb2 = np.stack([rgb2, rgb2, rgb2], axis=-1)
    return rgb2[..., ::-1].astype(np.uint8)


@dataclass
class TrackState:
    last_center: Tuple[float, float]
    first_seen_at: float
    last_moved_at: float
    last_seen_at: float
    movement_events: int


def _center(xyxy: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


@dataclass(frozen=True)
class Detection:
    xyxy: Tuple[int, int, int, int]
    conf: float


@dataclass(frozen=True)
class PersonView:
    track_id: int
    conf: float
    status_text: str
    seen_for_s: float
    accent_bgr: Tuple[int, int, int]
    xyxy: Tuple[int, int, int, int]


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _nms(dets: List[Detection], iou_thr: float) -> List[Detection]:
    dets_sorted = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Detection] = []
    for d in dets_sorted:
        suppressed = False
        for k in keep:
            if _iou(d.xyxy, k.xyxy) > iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


def _letterbox(
    img_bgr: np.ndarray, new_size: int, color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr, 1.0, (0, 0)
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    resized = _resize_bgr(img_bgr, new_unpad)
    dw = new_size - new_unpad[0]
    dh = new_size - new_unpad[1]
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top
    if cv2 is not None:
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
    else:
        padded = np.full((new_size, new_size, 3), color, dtype=resized.dtype)
        ph, pw = resized.shape[:2]
        padded[top : top + ph, left : left + pw] = resized
    return padded, r, (left, top)


def _scale_xyxy_back(
    xyxy: Tuple[float, float, float, float],
    ratio: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / ratio
    y1 = (y1 - pad_y) / ratio
    x2 = (x2 - pad_x) / ratio
    y2 = (y2 - pad_y) / ratio
    oh, ow = orig_shape
    x1i = int(max(0, min(ow - 1, round(x1))))
    y1i = int(max(0, min(oh - 1, round(y1))))
    x2i = int(max(0, min(ow, round(x2))))
    y2i = int(max(0, min(oh, round(y2))))
    return (x1i, y1i, x2i, y2i)


class _SimpleTracker:
    def __init__(self, match_iou: float, max_age_s: float) -> None:
        self._match_iou = match_iou
        self._max_age_s = max_age_s
        self._next_id = 1
        self._tracks: Dict[int, Tuple[Tuple[int, int, int, int], float]] = {}

    def update(self, detections: List[Detection], now: float) -> Dict[int, Detection]:
        for tid, (_, last_seen) in list(self._tracks.items()):
            if (now - last_seen) > self._max_age_s:
                del self._tracks[tid]

        assigned: Dict[int, Detection] = {}
        used_det = set()
        track_items = list(self._tracks.items())

        for tid, (tbox, _) in track_items:
            best_j = -1
            best_iou = 0.0
            for j, det in enumerate(detections):
                if j in used_det:
                    continue
                iou_val = _iou(tbox, det.xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0 and best_iou >= self._match_iou:
                det = detections[best_j]
                used_det.add(best_j)
                self._tracks[tid] = (det.xyxy, now)
                assigned[tid] = det

        for j, det in enumerate(detections):
            if j in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = (det.xyxy, now)
            assigned[tid] = det

        return assigned


class _OnnxYoloV8:
    def __init__(self, model_path: str) -> None:
        ort = _try_import_onnxruntime()
        if ort is None:
            raise RuntimeError("onnxruntime tidak tersedia. Install: pip install onnxruntime")
        self._ort = ort
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

    def detect(self, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> List[Detection]:
        img, ratio, pad = _letterbox(frame_bgr, imgsz)
        img_rgb = img[..., ::-1]
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        y = self._sess.run([self._output_name], {self._input_name: x})[0]

        if y.ndim == 3 and y.shape[0] == 1:
            y = y[0]
        if y.shape[0] < y.shape[1]:
            preds = y.T
        else:
            preds = y

        if preds.shape[1] < 5:
            return []

        cls_scores = preds[:, 4:]
        person_scores = cls_scores[:, 0] if cls_scores.shape[1] >= 1 else np.zeros((preds.shape[0],))
        keep = person_scores >= conf
        preds = preds[keep]
        person_scores = person_scores[keep]
        if preds.shape[0] == 0:
            return []

        xywh = preds[:, 0:4]
        cx = xywh[:, 0]
        cy = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        dets: List[Detection] = []
        oh, ow = frame_bgr.shape[:2]
        for i in range(x1.shape[0]):
            box = _scale_xyxy_back(
                (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                ratio=ratio,
                pad=pad,
                orig_shape=(oh, ow),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            dets.append(Detection(xyxy=box, conf=float(person_scores[i])))

        return _nms(dets, iou_thr=iou)


class _TfliteYoloV8:
    def __init__(self, model_path: str) -> None:
        Interpreter = _try_import_tflite()
        if Interpreter is None:
            raise RuntimeError(
                "tflite interpreter tidak tersedia. Install: pip install tflite-runtime (atau tensorflow)"
            )
        self._interp = Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._output = self._interp.get_output_details()[0]

    def detect(self, frame_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> List[Detection]:
        img, ratio, pad = _letterbox(frame_bgr, imgsz)
        img_rgb = img[..., ::-1]
        input_dtype = self._input["dtype"]
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        if input_dtype != np.float32:
            scale, zero = self._input.get("quantization", (1.0, 0))
            if scale and scale > 0:
                xq = (x / scale + zero).round()
                x = np.clip(xq, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
            else:
                x = x.astype(input_dtype)

        self._interp.set_tensor(self._input["index"], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._output["index"])

        if y.ndim == 3 and y.shape[0] == 1:
            y = y[0]
        if y.shape[0] < y.shape[1]:
            preds = y.T
        else:
            preds = y

        if preds.shape[1] < 5:
            return []

        cls_scores = preds[:, 4:]
        person_scores = cls_scores[:, 0] if cls_scores.shape[1] >= 1 else np.zeros((preds.shape[0],))
        keep = person_scores >= conf
        preds = preds[keep]
        person_scores = person_scores[keep]
        if preds.shape[0] == 0:
            return []

        xywh = preds[:, 0:4]
        cx = xywh[:, 0]
        cy = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        dets: List[Detection] = []
        oh, ow = frame_bgr.shape[:2]
        for i in range(x1.shape[0]):
            box = _scale_xyxy_back(
                (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                ratio=ratio,
                pad=pad,
                orig_shape=(oh, ow),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            dets.append(Detection(xyxy=box, conf=float(person_scores[i])))

        return _nms(dets, iou_thr=iou)


def _read_frame_termux_photo(camera_index: int, tmp_path: str) -> Optional[np.ndarray]:
    try:
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    except Exception:
        pass
    try:
        subprocess.run(
            ["termux-camera-photo", "-c", str(camera_index), tmp_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    if cv2 is not None:
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        return img
    Image = _try_import_pil_image()
    if Image is None:
        return None
    try:
        im = Image.open(tmp_path).convert("RGB")
        rgb = np.array(im)
        return rgb[..., ::-1].astype(np.uint8)
    except Exception:
        return None


def _format_people_count(count: int) -> str:
    return f"{int(count)} orang"


def _format_duration(seconds: float) -> str:
    seconds_i = max(0, int(seconds))
    mins, secs = divmod(seconds_i, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _overlay_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill_bgr: Tuple[int, int, int],
    alpha: float,
    border_bgr: Optional[Tuple[int, int, int]] = None,
    border_thickness: int = 1,
) -> None:
    if cv2 is None:
        return
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2].copy()
    panel = np.full_like(roi, fill_bgr, dtype=np.uint8)
    blended = cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0.0)
    frame[y1:y2, x1:x2] = blended
    if border_bgr is not None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_bgr, border_thickness, cv2.LINE_AA)


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    if cv2 is None:
        return
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_person_row(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    person: PersonView,
) -> int:
    row_h = 38
    _overlay_rect(frame, x, y, x + w, y + row_h, (23, 19, 36), 0.42, (62, 58, 90))
    _overlay_rect(frame, x + 8, y + 9, x + 20, y + row_h - 9, person.accent_bgr, 0.95, person.accent_bgr)
    _put_text(frame, f"#{person.track_id}  {person.status_text}", (x + 28, y + 18), 0.44, (242, 244, 252), 1)
    _put_text(frame, f"pantau { _format_duration(person.seen_for_s) }", (x + 28, y + 32), 0.36, (176, 184, 220), 1)
    if w >= 240:
        _put_text(frame, f"{person.conf:.2f}", (x + w - 48, y + 25), 0.42, (154, 235, 255), 1)
    return row_h


def _draw_hud(
    frame: np.ndarray,
    backend: str,
    source: str,
    people_in_frame: int,
    unique_count: int,
    peak_people_count: int,
    new_people_since_last_report: int,
    fps: float,
    uptime_s: float,
    active_people: List[PersonView],
) -> np.ndarray:
    if cv2 is None:
        return frame

    h, w = frame.shape[:2]
    compact = w < 1100 or h < 720
    header_h = 62 if compact else 72
    header_w = min(w - 24, 520 if compact else 620)
    _overlay_rect(frame, 12, 12, 12 + header_w, 12 + header_h, (14, 14, 24), 0.34, (72, 74, 102))
    _put_text(frame, "HumanDetection AI", (24, 36 if compact else 40), 0.72 if compact else 0.84, (248, 250, 255), 2)
    summary = (
        f"Sekarang {_format_people_count(people_in_frame)}   "
        f"Total {_format_people_count(unique_count)}   "
        f"Puncak {_format_people_count(peak_people_count)}"
    )
    _put_text(frame, summary, (24, 58 if compact else 66), 0.46 if compact else 0.52, (188, 194, 226), 1)

    if w >= 1150:
        tag_w = 300
        _overlay_rect(frame, w - tag_w - 12, 12, w - 12, 46, (16, 16, 28), 0.38, (72, 74, 102))
        _put_text(
            frame,
            f"{backend} | {source} | {fps:.1f} FPS | up {_format_duration(uptime_s)}",
            (w - tag_w, 35),
            0.42,
            (214, 218, 244),
            1,
        )
    elif w >= 880:
        _put_text(
            frame,
            f"{fps:.1f} FPS  |  up {_format_duration(uptime_s)}  |  +{new_people_since_last_report}",
            (24, 82 if compact else 92),
            0.42,
            (214, 218, 244),
            1,
        )

    if not active_people:
        return frame

    visible_rows = max(2, min(5 if compact else 6, (h - 180) // 44))
    panel_w = min(300 if compact else 340, max(210, int(w * (0.23 if compact else 0.26))))
    panel_h = 42 + visible_rows * 42
    panel_x1 = w - panel_w - 12
    panel_y1 = max(86, h - panel_h - 12)
    _overlay_rect(frame, panel_x1, panel_y1, w - 12, h - 12, (16, 16, 28), 0.34, (68, 70, 98))
    _put_text(frame, f"Orang Aktif ({len(active_people)})", (panel_x1 + 12, panel_y1 + 24), 0.52, (246, 248, 255), 1)

    row_y = panel_y1 + 34
    for person in active_people[:visible_rows]:
        row_h = _draw_person_row(frame, panel_x1 + 10, row_y, panel_w - 20, person)
        row_y += row_h + 4
        if row_y >= h - 16:
            break

    return frame


def run(
    backend: str,
    source: str,
    model_path: str,
    camera_index: int,
    conf: float,
    iou: float,
    imgsz: int,
    show_gui: bool,
    print_every: float,
    movement_px: float,
    movement_cooldown_s: float,
    track_iou: float,
    track_max_age_s: float,
    termux_interval_s: float,
    device: Optional[str],
    save_path: Optional[str],
) -> int:
    backend = (backend or "auto").strip().lower()
    source = (source or "opencv").strip().lower()

    if cv2 is None:
        if source == "opencv":
            sys.stderr.write("OpenCV tidak tersedia. Pakai --source termux-photo atau install opencv-python.\n")
            return 2
        if show_gui:
            sys.stderr.write("GUI butuh OpenCV. Jalankan dengan --no-gui.\n")
            return 2
        if save_path:
            sys.stderr.write("Opsi --save butuh OpenCV.\n")
            return 2

    model_ultra = None
    model_onnx = None
    model_tflite = None
    tracker = _SimpleTracker(match_iou=track_iou, max_age_s=track_max_age_s)

    if backend in ("auto", "ultralytics"):
        YOLO = _try_import_ultralytics()
        if YOLO is not None:
            try:
                model_ultra = YOLO(model_path)
            except Exception:
                model_ultra = None
        if backend == "ultralytics" and model_ultra is None:
            sys.stderr.write(
                "Backend ultralytics dipilih, tapi paket/model tidak bisa dipakai.\n"
                "Install: pip install ultralytics\n"
            )
            return 2

    if model_ultra is None and backend in ("auto", "onnx"):
        try:
            model_onnx = _OnnxYoloV8(model_path)
        except Exception:
            model_onnx = None
        if backend == "onnx" and model_onnx is None:
            sys.stderr.write(
                "Backend onnx dipilih, tapi onnxruntime/model tidak bisa dipakai.\n"
                "Install: pip install onnxruntime\n"
            )
            return 2

    if model_ultra is None and model_onnx is None and backend in ("auto", "tflite"):
        try:
            model_tflite = _TfliteYoloV8(model_path)
        except Exception:
            model_tflite = None
        if backend == "tflite" and model_tflite is None:
            sys.stderr.write(
                "Backend tflite dipilih, tapi interpreter/model tidak bisa dipakai.\n"
                "Install: pip install tflite-runtime (atau tensorflow)\n"
            )
            return 2

    if model_ultra is None and model_onnx is None and model_tflite is None:
        sys.stderr.write(
            "Tidak ada backend yang bisa dipakai.\n"
            "Opsi:\n"
            " - ultralytics (pt): pip install ultralytics\n"
            " - onnxruntime (onnx): pip install onnxruntime\n"
            " - tflite (tflite): pip install tflite-runtime\n"
        )
        return 2

    cap = None
    if source == "opencv":
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            sys.stderr.write(f"Tidak bisa membuka camera index={camera_index}\n")
            return 2

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 25.0
        width = 640
        height = 480
        if cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or np.isnan(fps) or fps < 1:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(save_path, fourcc, float(fps), (width, height))

    seen_ids = set()
    tracks: Dict[int, TrackState] = {}
    last_print = 0.0
    last_event_print: Dict[int, float] = {}
    last_reported_unique_count = 0
    session_started_at = time.time()
    last_frame_at = session_started_at
    fps_ema = 0.0
    peak_people_count = 0
    termux_tmp = "/data/data/com.termux/files/usr/tmp/human_detect_frame.jpg"

    try:
        while True:
            frame = None
            if source == "opencv":
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    sys.stderr.write("Gagal membaca frame dari camera.\n")
                    break
            elif source == "termux-photo":
                frame = _read_frame_termux_photo(camera_index=camera_index, tmp_path=termux_tmp)
                if frame is None:
                    sys.stderr.write(
                        "Gagal ambil foto dari termux-camera-photo. Pastikan termux-api terinstall.\n"
                    )
                    return 2
                if termux_interval_s > 0:
                    time.sleep(termux_interval_s)
            else:
                sys.stderr.write("source tidak dikenal. Pakai: opencv atau termux-photo\n")
                return 2

            now = time.time()
            dt = max(1e-6, now - last_frame_at)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema <= 0 else (fps_ema * 0.88 + inst_fps * 0.12)
            last_frame_at = now
            people_in_frame = 0
            active_people: List[PersonView] = []

            annotated = frame
            per_det_conf: Dict[int, float] = {}
            if model_ultra is not None:
                results = model_ultra.track(
                    source=frame,
                    persist=True,
                    classes=[0],
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )
                if results and len(results) > 0:
                    r0 = results[0]
                    boxes = getattr(r0, "boxes", None)
                    if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32)
                        ids = None
                        if getattr(boxes, "id", None) is not None:
                            ids = boxes.id.detach().cpu().numpy().astype(np.int32)
                        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
                        for i in range(len(xyxy)):
                            track_id = int(ids[i]) if ids is not None else -1
                            if track_id >= 0:
                                per_det_conf[track_id] = float(confs[i]) if confs is not None else 0.0
                        det_map = {}
                        for i in range(len(xyxy)):
                            box = tuple(map(int, xyxy[i].tolist()))
                            track_id = int(ids[i]) if ids is not None else -1
                            if track_id >= 0:
                                det_map[track_id] = Detection(xyxy=box, conf=per_det_conf.get(track_id, 0.0))
                        tracked = det_map
                    else:
                        tracked = {}
                else:
                    tracked = {}
            else:
                if model_onnx is not None:
                    dets = model_onnx.detect(frame, conf=conf, iou=iou, imgsz=imgsz)
                else:
                    assert model_tflite is not None
                    dets = model_tflite.detect(frame, conf=conf, iou=iou, imgsz=imgsz)
                tracked = tracker.update(dets, now=now)
                for tid, det in tracked.items():
                    per_det_conf[tid] = float(det.conf)

            for track_id, det in tracked.items():
                people_in_frame += 1
                box = det.xyxy
                seen_ids.add(track_id)

                cx, cy = _center(box)
                moved = False
                prev = tracks.get(track_id)
                if prev is None:
                    tracks[track_id] = TrackState(
                        last_center=(cx, cy),
                        first_seen_at=now,
                        last_moved_at=0.0,
                        last_seen_at=now,
                        movement_events=0,
                    )
                    current_state = tracks[track_id]
                else:
                    dist = _distance(prev.last_center, (cx, cy))
                    if dist >= movement_px and (now - prev.last_moved_at) >= movement_cooldown_s:
                        moved = True
                        prev.last_moved_at = now
                        prev.movement_events += 1
                    prev.last_center = (cx, cy)
                    prev.last_seen_at = now
                    current_state = prev

                if track_id >= 0 and moved:
                    last_t = last_event_print.get(track_id, 0.0)
                    if (now - last_t) >= 0.15:
                        sys.stdout.write(
                            f"event=bergerak id={track_id} conf={per_det_conf.get(track_id, 0.0):.2f}\n"
                        )
                        last_event_print[track_id] = now

                if moved:
                    status_text = "Bergerak"
                elif current_state.movement_events > 0:
                    status_text = "Aktif"
                else:
                    status_text = "Dipantau"

                if status_text == "Bergerak":
                    accent_bgr = (0, 200, 255)
                elif status_text == "Aktif":
                    accent_bgr = (0, 255, 170)
                else:
                    accent_bgr = (255, 190, 70)

                active_people.append(
                    PersonView(
                        track_id=track_id,
                        conf=per_det_conf.get(track_id, 0.0),
                        status_text=status_text,
                        seen_for_s=now - current_state.first_seen_at,
                        accent_bgr=accent_bgr,
                        xyxy=box,
                    )
                )

                if show_gui:
                    label = f"#{track_id}  {status_text}"
                    x1, y1, x2, y2 = box
                    _overlay_rect(annotated, x1, y1, x2, y2, accent_bgr, 0.04, accent_bgr, 2)
                    label_w = min(140, max(88, (x2 - x1)))
                    text_y1 = max(6, y1 - 28)
                    text_y2 = max(28, y1 - 4)
                    _overlay_rect(annotated, x1, text_y1, x1 + label_w, text_y2, (20, 20, 32), 0.52, accent_bgr)
                    _put_text(annotated, label, (x1 + 8, text_y2 - 8), 0.42, (250, 250, 255), 1)

            active_people.sort(key=lambda p: p.track_id)
            unique_count = len(seen_ids)
            peak_people_count = max(peak_people_count, people_in_frame)
            new_people_since_last_report = max(0, unique_count - last_reported_unique_count)

            if now - last_print >= print_every:
                sys.stdout.write(
                    f"terlihat_sekarang={_format_people_count(people_in_frame)} | "
                    f"total_terdeteksi={_format_people_count(unique_count)} | "
                    f"puncak={_format_people_count(peak_people_count)} | "
                    f"fps={fps_ema:.1f} | "
                    f"uptime={_format_duration(now - session_started_at)} | "
                    f"orang_baru=+{new_people_since_last_report}\n"
                )
                for person in active_people:
                    sys.stdout.write(
                        f" - Orang #{person.track_id} | akurasi={person.conf:.2f} | "
                        f"status={person.status_text} | pantau={_format_duration(person.seen_for_s)}\n"
                    )
                sys.stdout.flush()
                last_print = now
                last_reported_unique_count = unique_count

            if show_gui:
                annotated = _draw_hud(
                    annotated,
                    backend=backend,
                    source=source,
                    people_in_frame=people_in_frame,
                    unique_count=unique_count,
                    peak_people_count=peak_people_count,
                    new_people_since_last_report=new_people_since_last_report,
                    fps=fps_ema,
                    uptime_s=now - session_started_at,
                    active_people=active_people,
                )
                cv2.imshow("Human Detection (YOLOv8n)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if writer is not None:
                writer.write(annotated)

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if show_gui:
            cv2.destroyAllWindows()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="human_detect_cli",
        description=(
            "Aplikasi terminal HumanDetection AI: deteksi manusia, tracking realtime, hitung otomatis, status gerak, dan dashboard monitoring modern."
        ),
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ultralytics", "onnx", "tflite"],
        help="Backend inference. auto akan coba ultralytics -> onnx -> tflite",
    )
    p.add_argument(
        "--source",
        default="opencv",
        choices=["opencv", "termux-photo"],
        help="Sumber frame. termux-photo pakai termux-camera-photo (polling).",
    )
    p.add_argument("--model", default="yolov8n.pt", help="Path model YOLOv8n (.pt)")
    p.add_argument("--camera", type=int, default=0, help="Index camera device (default 0)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Ukuran input model (default 640)")
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Jalankan tanpa jendela preview (cocok untuk Termux/headless)",
    )
    p.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Interval cetak status ke terminal (detik)",
    )
    p.add_argument(
        "--movement-px",
        type=float,
        default=18.0,
        help="Ambang jarak (px) agar dianggap bergerak",
    )
    p.add_argument(
        "--movement-cooldown",
        type=float,
        default=0.6,
        help="Minimal jeda deteksi gerak per ID (detik)",
    )
    p.add_argument(
        "--track-iou",
        type=float,
        default=0.30,
        help="IoU minimal untuk match tracking (onnx/tflite)",
    )
    p.add_argument(
        "--track-max-age",
        type=float,
        default=1.5,
        help="Berapa lama track boleh hilang sebelum dihapus (detik) (onnx/tflite)",
    )
    p.add_argument(
        "--termux-interval",
        type=float,
        default=0.10,
        help="Jeda antar foto (detik) saat source=termux-photo",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device ultralytics: 'cpu', '0' (gpu), dsb. Default auto",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Simpan video hasil anotasi (mp4). Contoh: --save out.mp4",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    return run(
        backend=args.backend,
        source=args.source,
        model_path=args.model,
        camera_index=args.camera,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show_gui=not args.no_gui,
        print_every=max(0.1, float(args.print_every)),
        movement_px=max(1.0, float(args.movement_px)),
        movement_cooldown_s=max(0.0, float(args.movement_cooldown)),
        track_iou=max(0.0, float(args.track_iou)),
        track_max_age_s=max(0.1, float(args.track_max_age)),
        termux_interval_s=max(0.0, float(args.termux_interval)),
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    raise SystemExit(main())
