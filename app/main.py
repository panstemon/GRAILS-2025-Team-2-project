# -*- coding: utf-8 -*-
"""
YOLOv5 + Face-Recognition Demo GUI (PyQt5)
• Shows each detected object in a green box.
• For class 0 (person) the label is replaced with the recognised face name,
  otherwise it stays “person”. Unknown faces keep the “person” label.
"""

import sys, pickle
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QWidget, QInputDialog, QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

# --------------------  FACE LIB (optional)  --------------------
try:
    import face_recognition                                  # type: ignore
    _FACE_OK = True
except ImportError:
    print("face_recognition not installed – face ID disabled")
    _FACE_OK = False

_FACE_DB = Path("known_faces.pkl")


def _load_faces() -> Tuple[List[str], List[np.ndarray]]:
    if _FACE_DB.exists():
        with _FACE_DB.open("rb") as f:
            d = pickle.load(f)
            return d.get("names", []), d.get("encodings", [])
    return [], []


def _save_faces(nm: List[str], enc: List[np.ndarray]):
    with _FACE_DB.open("wb") as f:
        pickle.dump({"names": nm, "encodings": enc}, f)


KNOWN_NAMES, KNOWN_ENCODINGS = _load_faces() if _FACE_OK else ([], [])

# --------------------  MODEL INIT  --------------------
print("Loading YOLOv5 model…")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = (
    torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    .to(_DEVICE)
    .eval()
)
if _DEVICE == "cuda":
    _MODEL.half()
print(f"Model loaded on {_DEVICE}")

ALLOWED_NAMES = {
    "person",
    "knife", "scissors", "baseball bat", "fork", "baseball glove", "tennis racket",
    "car", "motorcycle", "truck", "bus", "train", "bicycle",
    "airplane", "traffic light", "stop sign",
}
ALLOWED_IDX = {i for i, n in _MODEL.names.items() if n in ALLOWED_NAMES}

# --------------------  GUI  ---------------------------
class DetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Object & Face Detection")
        self.setGeometry(100, 100, 800, 700)

        self.preview = QLabel(alignment=0x84)        # center
        self.preview.setFixedSize(640, 480)

        self.cam_box = QComboBox(); self.cam_box.addItems([str(i) for i in range(5)])

        self.btn_start  = QPushButton("Start Camera")
        self.btn_video  = QPushButton("Open Video…")
        self.btn_stop   = QPushButton("Stop")
        self.btn_addfac = QPushButton("Add Person…"); self.btn_addfac.setEnabled(_FACE_OK)
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_addfac.clicked.connect(self.add_person)

        lay = QVBoxLayout()
        for w in (
            self.preview, self.cam_box,
            self.btn_start, self.btn_video, self.btn_stop, self.btn_addfac
        ):
            lay.addWidget(w)
        c = QWidget(); c.setLayout(lay); self.setCentralWidget(c)

        self.cap = None
        self.timer = QTimer(self); self.timer.timeout.connect(self.next_frame)
        self.is_file = False

    # ---------- source handling ----------
    def start_camera(self):
        self._start(cv2.VideoCapture(int(self.cam_box.currentText())), is_file=False)

    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if p: self._start(cv2.VideoCapture(p), is_file=True)

    def _start(self, cap: cv2.VideoCapture, *, is_file: bool):
        self.stop_stream()
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open source"); return
        self.cap, self.is_file = cap, is_file
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.btn_start.setEnabled(False); self.btn_video.setEnabled(False); self.btn_stop.setEnabled(True)
        self.timer.start(30); print("Stream started")

    def stop_stream(self):
        if self.cap:
            self.timer.stop(); self.cap.release(); self.cap = None
            self.preview.clear()
            self.btn_start.setEnabled(True); self.btn_video.setEnabled(True); self.btn_stop.setEnabled(False)
            print("Stream stopped")

    # ---------- face DB ----------
    def add_person(self):
        if not _FACE_OK: return
        imgs, _ = QFileDialog.getOpenFileNames(self, "Select face images", "", "Images (*.jpg *.png *.jpeg)")
        if not imgs: return
        name, ok = QInputDialog.getText(self, "Person Name", "Enter name:"); name = name.strip()
        if not ok or not name: return
        new_encs = []
        for p in imgs:
            img = face_recognition.load_image_file(p)
            locs = face_recognition.face_locations(img, model="hog")
            if not locs: continue
            new_encs.append(face_recognition.face_encodings(img, locs)[0])
        if not new_encs:
            QMessageBox.information(self, "Face Add", "No faces found."); return
        global KNOWN_NAMES, KNOWN_ENCODINGS
        KNOWN_NAMES.extend([name]*len(new_encs)); KNOWN_ENCODINGS.extend(new_encs)
        _save_faces(KNOWN_NAMES, KNOWN_ENCODINGS)
        QMessageBox.information(self, "Face Add", f"Added {len(new_encs)} image(s) for {name}.")

    # ---------- main loop ----------
    def next_frame(self):
        if not (self.cap and self.cap.isOpened()):
            self.stop_stream(); return
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file: print("Video finished")
            self.stop_stream(); return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _MODEL(img_rgb, size=640)
        pred = res.pred[0]

        # filter unwanted classes
        if pred is not None and pred.shape[0]:
            keep = torch.tensor([int(c) in ALLOWED_IDX for c in pred[:,5]], dtype=torch.bool, device=pred.device)
            pred = pred[keep]

        annotated = frame.copy()

        # build list of detections
        boxes = pred.cpu().numpy() if pred is not None else np.empty((0,6))
        labels = []
        for *xyxy, conf, cls in boxes:
            cls = int(cls)
            labels.append(_MODEL.names[cls])

        # ------ FACE RECOG to rename person boxes ------
        if boxes.size and _FACE_OK and KNOWN_ENCODINGS:
            shrink = 0.25 if min(img_rgb.shape[:2]) > 720 else 0.8
            small  = cv2.resize(img_rgb, (0,0), fx=shrink, fy=shrink)
            locs   = face_recognition.face_locations(small, number_of_times_to_upsample=2, model="hog")
            encs   = face_recognition.face_encodings(small, locs)
            for (top, right, bottom, left), enc in zip(locs, encs):
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS, enc, tolerance=0.5)
                if True not in matches: continue
                # majority vote
                counts = {}
                for i,m in enumerate(matches):
                    if m: counts[KNOWN_NAMES[i]] = counts.get(KNOWN_NAMES[i],0)+1
                name = max(counts, key=counts.get)

                # upscale to full frame
                s = 1/shrink
                cx = int(((left+right)*0.5)*s); cy = int(((top+bottom)*0.5)*s)
                # find person box that encloses this face centre
                for i, (*xyxy, conf, cls) in enumerate(boxes):
                    if int(cls)!=0: continue
                    x1,y1,x2,y2 = map(int,xyxy)
                    if x1<=cx<=x2 and y1<=cy<=y2:
                        labels[i] = name               # replace label
                        break

        # ------ draw everything ------
        for (x1,y1,x2,y2,conf,cls), lbl in zip(boxes, labels):
            x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(annotated,(x1,y1-20),(x1+len(lbl)*11,y1),(0,255,0),-1)
            cv2.putText(annotated,lbl,(x1+2,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1)

        # Qt preview
        h,w,ch = annotated.shape
        img = QImage(annotated.data,w,h,ch*w,QImage.Format_BGR888)
        self.preview.setPixmap(QPixmap.fromImage(img))

# --------------------  MAIN  --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DetectorGUI(); gui.show()
    sys.exit(app.exec_())
