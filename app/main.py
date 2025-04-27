# -*- coding: utf-8 -*-
"""
YOLOv5 + Face‑Recognition Demo GUI        (PyQt5)
───────────────────────────────────────────
• Realtime object detection with Ultralytics YOLOv5 (USB cam / video file)
• Plug‑and‑play face recognition:
      – Press “Add Person” → choose arbitrary face photos → enter a name.
      – Faces are encoded once and stored to *known_faces.pkl* (persistent).
      – At runtime every detected face is matched and overlaid with the name.
Optimisations
─────────────
* The YOLO model is loaded once, moved to GPU if available, and set to FP16.
* A single‑threaded Qt timer (30 ms) updates the preview without extra threads.
* face_recognition encodings are cached; heavy ops run only when needed.
Requirements
────────────
$ pip install pyqt5 opencv‑python torch torchvision torchaudio
$ pip install face_recognition      # Wraps dlib → see Windows build notes
If *face_recognition* is not available the GUI still runs (only YOLO part).
"""

import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QVBoxLayout,
    QWidget,
    QInputDialog,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

import warnings
warnings.filterwarnings("ignore", 
    message=".*torch.cuda.amp.autocast.*deprecated.*", 
    category=FutureWarning)

# --------------------  FACE LIB (optional)  -------------------- #
try:
    import face_recognition  # type: ignore

    _FACE_OK = True
except ImportError:
    print("face_recognition not installed – face ID disabled")
    _FACE_OK = False

_FACE_DB = Path("known_faces.pkl")


def _load_faces() -> Tuple[List[str], List[np.ndarray]]:
    if _FACE_DB.exists():
        with _FACE_DB.open("rb") as f:
            data = pickle.load(f)
        return data.get("names", []), data.get("encodings", [])
    return [], []


def _save_faces(names: List[str], encodings: List[np.ndarray]):
    with _FACE_DB.open("wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)


KNOWN_NAMES, KNOWN_ENCODINGS = _load_faces() if _FACE_OK else ([], [])

# --------------------  MODEL INIT  -------------------- #
print("Loading YOLOv5 model…")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    .to(_DEVICE)
    .eval()
)
if _DEVICE == "cuda":
    _MODEL.half()  # FP16 for extra speed
print(f"Model loaded on {_DEVICE}")


# --------------------  GUI  --------------------------- #
class DetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Object & Face Detection")
        self.setGeometry(100, 100, 800, 700)

        # Preview widget
        self.preview = QLabel(alignment=0x84)  # Align center
        self.preview.setFixedSize(640, 480)

        # Camera selector (IDs 0‑5 are usually enough)
        self.cam_box = QComboBox(); self.cam_box.addItems([str(i) for i in range(5)])

        # Buttons
        self.btn_start  = QPushButton("Start Camera")
        self.btn_video  = QPushButton("Open Video…")
        self.btn_stop   = QPushButton("Stop")
        self.btn_addfac = QPushButton("Add Person…")
        self.btn_addfac.setEnabled(_FACE_OK)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_addfac.clicked.connect(self.add_person)
        self.btn_stop.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        for w in (
            self.preview,
            self.cam_box,
            self.btn_start,
            self.btn_video,
            self.btn_stop,
            self.btn_addfac,
        ):
            layout.addWidget(w)
        container = QWidget(); container.setLayout(layout); self.setCentralWidget(container)

        # Runtime vars
        self.cap = None
        self.timer = QTimer(self); self.timer.timeout.connect(self.next_frame)
        self.is_file = False

    # ---------- SOURCES ---------- #

    def start_camera(self):
        cam_id = int(self.cam_box.currentText())
        self._start(cv2.VideoCapture(cam_id), is_file=False)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._start(cv2.VideoCapture(path), is_file=True)

    def _start(self, capture: cv2.VideoCapture, *, is_file: bool):
        self.stop_stream()
        if not capture.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open source")
            return
        self.cap, self.is_file = capture, is_file
        # Reduce buffering latency on webcams
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.btn_start.setEnabled(False); self.btn_video.setEnabled(False); self.btn_stop.setEnabled(True)
        self.timer.start(30)
        print("Stream started")

    def stop_stream(self):
        if self.cap:
            self.timer.stop(); self.cap.release(); self.cap = None
            self.preview.clear()
            self.btn_start.setEnabled(True); self.btn_video.setEnabled(True); self.btn_stop.setEnabled(False)
            print("Stream stopped")

    # ---------- FACE TRAINING ---------- #

    def add_person(self):
        if not _FACE_OK:
            return
        images, _ = QFileDialog.getOpenFileNames(self, "Select face images", "", "Images (*.jpg *.png *.jpeg)")
        if not images:
            return
        name, ok = QInputDialog.getText(self, "Person Name", "Enter the person's name:")
        if not ok or not name.strip():
            return
        new_encs = []
        for img_path in images:
            img = face_recognition.load_image_file(img_path)
            locs = face_recognition.face_locations(img, model="hog")
            if not locs:
                print(f"No face in {img_path}")
                continue
            enc = face_recognition.face_encodings(img, locs)[0]
            new_encs.append(enc)
        if not new_encs:
            QMessageBox.information(self, "Face Add", "No valid faces found in selected images.")
            return
        global KNOWN_NAMES, KNOWN_ENCODINGS
        KNOWN_NAMES.extend([name] * len(new_encs))
        KNOWN_ENCODINGS.extend(new_encs)
        _save_faces(KNOWN_NAMES, KNOWN_ENCODINGS)
        QMessageBox.information(self, "Face Add", f"Added {len(new_encs)} image(s) for {name}.")

    # ----- FRAME LOOP OPTIMIZATION ----- #

    def next_frame(self):
        if not (self.cap and self.cap.isOpened()):
            self.stop_stream(); return

        ok, frame = self.cap.read()
        if not ok:
            if self.is_file:
                print("Video finished")
            self.stop_stream(); return

        # Resize frame early to minimize overhead
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert color once for all operations
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # YOLOv5 (small inference size 320)
        results = _MODEL(img_rgb, size=320)

        # Manual lightweight YOLO rendering
        annotated = frame_resized.copy()
        pred = results.xyxy[0].cpu().numpy()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            label = results.names[int(cls)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Reduce face recognition frequency
        if _FACE_OK and KNOWN_ENCODINGS and (self.timer.remainingTime() % 90 < 30):
            small = cv2.resize(img_rgb, (0, 0), fx=0.25, fy=0.25)
            locs = face_recognition.face_locations(small, model="hog")
            encs = face_recognition.face_encodings(small, locs)

            for (top, right, bottom, left), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS, enc, tolerance=0.55)
                name = "Unknown"
                if any(matches):
                    matched_idx = matches.index(True)
                    name = KNOWN_NAMES[matched_idx]
                top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
                cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(annotated, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Qt preview (optimized)
        qt_img = QImage(annotated.data, annotated.shape[1], annotated.shape[0],
                        annotated.shape[1] * 3, QImage.Format_BGR888)
        self.preview.setPixmap(QPixmap.fromImage(qt_img))


# --------------------  MAIN  -------------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DetectorGUI(); gui.show()
    sys.exit(app.exec_())
