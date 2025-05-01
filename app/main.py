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

# ---------------------------------------------------------------
# Put this near the top of your file, after the model loads
ALLOWED_NAMES = {
    "person",
    "knife", "scissors", "baseball bat", "fork", "baseball glove", "tennis racket",
    "car", "motorcycle", "truck", "bus", "train", "bicycle",
    "airplane", "traffic light", "stop sign",
}
ALLOWED_IDX = {i for i, n in _MODEL.names.items() if n in ALLOWED_NAMES}
# ---------------------------------------------------------------


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

    # ---------- FRAME LOOP ---------- #

    def next_frame(self):
        if not (self.cap and self.cap.isOpened()):
            self.stop_stream(); return
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file:
                print("Video finished")
            self.stop_stream(); return

        # ---------- YOLO OBJECT DETECTION ---------- #
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _MODEL(img_rgb, size=640)

        # --- keep only “allowed” detections ---
        pred = results.pred[0]                      # tensor [N, 6]  (x1,y1,x2,y2,conf,cls)
        if pred is not None and pred.shape[0]:
            keep_mask = torch.tensor(
                [int(c) in ALLOWED_IDX for c in pred[:, 5]],
                dtype=torch.bool,
                device=pred.device,
            )
            results.pred[0] = pred[keep_mask]       # ← *in‑place* pruning

        # Person flag for face‑ID
        person_detected = 0 in pred[:, 5] if pred is not None and pred.shape[0] else False

        # Draw only the remaining boxes
        annotated = results.render()[0].copy()
        

        # ---------- FACE RECOGNITION ---------- #
        if person_detected and _FACE_OK and KNOWN_ENCODINGS:
            shrink = 0.25 if min(img_rgb.shape[:2]) > 720 else 0.8
            small  = cv2.resize(img_rgb, (0, 0), fx=shrink, fy=shrink)  # speed
            locs = face_recognition.face_locations(small,  number_of_times_to_upsample=2, model="hog")
            encs = face_recognition.face_encodings(small, locs)
            for (top, right, bottom, left), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS, enc, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    idxs = [i for i, m in enumerate(matches) if m]
                    counts = {}
                    for i in idxs:
                        counts[KNOWN_NAMES[i]] = counts.get(KNOWN_NAMES[i], 0) + 1
                    name = max(counts, key=counts.get)

                if name == "Unknown":
                    continue
                # Scale back up face locations since we detected on the resized image
                scale  = 1 / shrink
                top, right, bottom, left = [int(v * scale) for v in (top, right, bottom, left)]

                cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(annotated, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(annotated, name, (left + 2, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ---------- Qt preview ---------- #
        h, w, ch = annotated.shape
        qt_img = QImage(annotated.data, w, h, ch * w, QImage.Format_BGR888)
        self.preview.setPixmap(QPixmap.fromImage(qt_img))


# --------------------  MAIN  -------------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DetectorGUI(); gui.show()
    sys.exit(app.exec_())