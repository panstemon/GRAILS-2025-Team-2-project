# -*- coding: utf-8 -*-
"""
Optimised YOLOv5 + Face-Recognition GUI (PyQt5)
───────────────────────────────────────────────
◆ Windows-friendly, no funky compiler steps.
◆ YOLO runs FP16 on CUDA, FP32 on CPU.
◆ Face recognition scoped to each detected person ROI – now using a fully
  contiguous RGB array and letting *face_recognition* detect landmarks itself
  (fixes the `compute_face_descriptor()` TypeError on some dlib wheels).
◆ Single Qt timer – zero threads, zero frame queue latency.

Install:
    pip install pyqt5 opencv-python torch torchvision torchaudio
    pip install face_recognition   # optional, enables face ID

Run:
    python yolo_face_gui_optimized.py
"""

import os, sys, pickle, time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QComboBox, QVBoxLayout, QWidget, QInputDialog, QTextEdit,        # ← added QTextEdit
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────  Torch & OpenCV tweaks  ────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # weird Windows + PyTorch edge-case
cv2.ocl.setUseOpenCL(False)                   # avoid OpenCL path on AMD/Intel iGPUs
try:
    import torch._dynamo as _dynamo; _dynamo.disable()  # skip Dynamo/Inductor
except Exception:
    pass

torch.set_grad_enabled(False)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────  Face DB helpers  ──────────────────────────────────
try:
    import face_recognition  # type: ignore
    _FACE_OK = True
except ImportError:
    print("face_recognition not installed – face ID disabled"); _FACE_OK = False

_FACE_DB = Path("known_faces.pkl")

def _load_faces() -> Tuple[List[str], List[np.ndarray]]:
    if _FACE_DB.exists():
        with _FACE_DB.open("rb") as f:
            data = pickle.load(f); return data["names"], data["encodings"]
    return [], []

def _save_faces(names: List[str], encs: List[np.ndarray]):
    with _FACE_DB.open("wb") as f:
        pickle.dump({"names": names, "encodings": encs}, f)

KNOWN_NAMES, KNOWN_ENCS = _load_faces() if _FACE_OK else ([], [])

# ─────────────────────────  YOLO initialisation  ─────────────────────────────
print("Loading YOLOv5 model …")
_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    .to(_DEVICE)
    .eval()
)
if _DEVICE == "cuda":
    _MODEL.half()
print(f"Model ready on {_DEVICE}")

ALLOWED_CLASSES = {
    "person", "knife", "scissors", "baseball bat", "fork", "baseball glove", "tennis racket",
    "car", "motorcycle", "truck", "bus", "train", "bicycle",
    "airplane", "traffic light", "stop sign",
}
ALLOWED_IDX = {i for i, n in _MODEL.names.items() if n in ALLOWED_CLASSES}

# Deterministic colour per class id (BGR)
_COLOURS = {}

def _colour(cls_id: int):
    if cls_id not in _COLOURS:
        rng = np.random.RandomState(cls_id * 13 + 17)
        _COLOURS[cls_id] = tuple(int(x) for x in rng.randint(50, 256, 3))
    return _COLOURS[cls_id]

# ─────────────────────────  Face-ID helper  ────────────────────────────
class _FaceID:
    """Lightweight cache + vectorised matcher (keeps main loop thread-free)."""
    EVERY_N = 4                 # run full encoding once every N GUI frames
    DOWNSCALE = 0.75            # shrink ROI edge; 1.0 = no downscale
    TOLERANCE = 0.48

    def __init__(self):
        self._frame_id = 0
        self._cache: dict[Tuple[int, int, int, int], str] = {}
        # pre-stack for fast broadcasting
        self._encs = np.vstack(KNOWN_ENCS) if KNOWN_ENCS else np.empty((0, 128))

    def name_for(self, rgb_roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> str | None:
        """Return cached or freshly computed name for the given bbox."""
        self._frame_id += 1

        # look-up cache (simple IoU test)
        bx1, by1, bx2, by2 = bbox
        for (cx1, cy1, cx2, cy2), cname in list(self._cache.items()):
            # Intersection-over-Union; >0.4 ⇒ “same face”
            inter = max(0, min(bx2, cx2) - max(bx1, cx1)) * max(0, min(by2, cy2) - max(by1, cy1))
            area1 = (bx2 - bx1) * (by2 - by1)
            area2 = (cx2 - cx1) * (cy2 - cy1)
            if inter / (area1 + area2 - inter + 1e-6) > 0.4:
                return cname                          # reuse previous result

        # not cached OR scheduled refresh
        if self._frame_id % self.EVERY_N:
            return None

        if self.DOWNSCALE != 1.0:
            rgb_roi = cv2.resize(rgb_roi, dsize=None, fx=self.DOWNSCALE, fy=self.DOWNSCALE)

        encs = face_recognition.face_encodings(rgb_roi, num_jitters=1)
        if not (encs and self._encs.size):
            return None

        dists = face_recognition.face_distance(self._encs, encs[0])
        best_idx = np.argmin(dists)
        if dists[best_idx] < self.TOLERANCE:
            name = KNOWN_NAMES[best_idx]
            self._cache[bbox] = name                 # memoise
            return name
        return None

_FACE_ID = _FaceID()

# ─────────────────────────  GUI  ──────────────────────────────────────────────
class DetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Object & Face Detection")
        self.setGeometry(100, 100, 820, 820)

        self.label_preview = QLabel(alignment=Qt.AlignCenter)
        self.label_preview.setFixedSize(640, 480)

        # ---------- NEW: live log box ----------
        self.log_box = QTextEdit(readOnly=True)
        self.log_box.setFixedHeight(140)
        self.log_box.setStyleSheet("font-family: Consolas;")

        self.combo_cam = QComboBox(); self.combo_cam.addItems([str(i) for i in range(5)])
        self.btn_start = QPushButton("Start Camera"); self.btn_video = QPushButton("Open Video …")
        self.btn_stop  = QPushButton("Stop");         self.btn_faces = QPushButton("Add Person …")
        self.btn_faces.setEnabled(_FACE_OK); self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._start_camera)
        self.btn_video.clicked.connect(self._open_video)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_faces.clicked.connect(self._add_person)

        lay = QVBoxLayout()
        for w in (
            self.label_preview, self.log_box,          # ← include log box
            self.combo_cam, self.btn_start, self.btn_video,
            self.btn_stop, self.btn_faces
        ):
            lay.addWidget(w)
        container = QWidget(); container.setLayout(lay); self.setCentralWidget(container)

        self.cap = None; self._file_mode = False
        self.timer = QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self._next)

    # --------------- helper to write to log ---------------
    def _log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    # ───────────────────  Source management  ───────────────────
    def _start_camera(self):
        self._start(cv2.VideoCapture(int(self.combo_cam.currentText())), is_file=False)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path: self._start(cv2.VideoCapture(path), is_file=True)

    def _start(self, cap: cv2.VideoCapture, *, is_file: bool):
        self._stop()
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open source"); return
        self.cap, self._file_mode = cap, is_file
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.log_box.clear()                                        # reset log
        for b in (self.btn_start, self.btn_video): b.setEnabled(False)
        self.btn_stop.setEnabled(True); self.timer.start(); print("Stream started")

    def _stop(self):
        if self.cap:
            self.timer.stop(); self.cap.release(); self.cap = None; self.label_preview.clear()
            for b in (self.btn_start, self.btn_video): b.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if _DEVICE == "cuda": torch.cuda.empty_cache()
            print("Stream stopped")

    # ───────────────────  Face DB actions  ────────────────────
    def _add_person(self):
        if not _FACE_OK: return
        files, _ = QFileDialog.getOpenFileNames(self, "Select face images", "", "Images (*.jpg *.png *.jpeg)")
        if not files: return
        name, ok = QInputDialog.getText(self, "Person Name", "Enter name:")
        name = name.strip()
        if not ok or not name: return
        new_encs: List[np.ndarray] = []
        for f in files:
            img = face_recognition.load_image_file(f)
            locs = face_recognition.face_locations(img, model="hog")
            if locs:
                new_encs.append(face_recognition.face_encodings(img, locs)[0])
        if not new_encs:
            QMessageBox.information(self, "Face Add", "No faces found."); return
        KNOWN_NAMES.extend([name] * len(new_encs)); KNOWN_ENCS.extend(new_encs)
        _save_faces(KNOWN_NAMES, KNOWN_ENCS)
        QMessageBox.information(self, "Face Add", f"Added {len(new_encs)} image(s) for {name}.")

    # ───────────────────  Main loop  ──────────────────────────
    def _next(self):
        if not (self.cap and self.cap.isOpened()):
            self._stop(); return
        ok, frame = self.cap.read()
        if not ok:
            if self._file_mode: print("Video finished")
            self._stop(); return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = _MODEL(img_rgb, size=640).pred[0]

        if pred is not None and pred.shape[0]:
            keep = torch.tensor([int(c) in ALLOWED_IDX for c in pred[:, 5]],
                                device=pred.device, dtype=torch.bool)
            pred = pred[keep]
        else:
            pred = torch.empty((0, 6))

        boxes = pred.cpu().numpy()
        labels = [_MODEL.names[int(cls)] for *_, cls in boxes]

        if _FACE_OK and len(boxes):
            for idx, (*xyxy, conf, cls) in enumerate(boxes):
                if int(cls) != 0:        # only ‘person’ boxes get a face-check
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                pad = int(0.1 * (y2 - y1))
                roi = img_rgb[max(0, y1 - pad): y2 + pad, max(0, x1 - pad): x2 + pad]
                if roi.size < 1024:
                    continue
                name = _FACE_ID.name_for(roi, (x1, y1, x2, y2))
                if name:
                    labels[idx] = name

        # draw + log
        if len(boxes):
            for (*xyxy, conf, cls), lbl in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, xyxy); colour = _colour(int(cls))
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
                cv2.putText(frame, lbl, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                self._log(f"{lbl} – {conf:.2f}")
        else:
            self._log("–")

        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        self.label_preview.setPixmap(QPixmap.fromImage(qimg))

# ─────────────────────────  Main entry  ───────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv); gui = DetectorGUI(); gui.show(); sys.exit(app.exec())
