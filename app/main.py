"""
Ultra‑minimal PyQt5 demo that runs YOLOv5 object detection on either:
  • a selected USB/web camera (choose ID from a dropdown)  
  • an .mp4 / .avi / .mov / .mkv video file

Optimisations:
  * Loads the model once, moves it to GPU if available, and switches to half‑precision (FP16) on CUDA.
  * Uses the built‑in `render()` method of Ultralytics YOLO for fastest NumPy → BGR annotation.
  * Resizes inference input to 640 px on its longer side for speed (adjustable).
  * Single‑threaded Qt timer loop (30 ms) keeps the GUI responsive without additional threads.
  * No TTS, face‑rec, or config files – just pure detection + drawing.
"""

import sys
import cv2
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
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

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

# --------------------  APP  --------------------------- #
class DetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Realtime Object Detection")
        self.setGeometry(100, 100, 800, 650)

        # Preview widget
        self.preview = QLabel(alignment=0x84)  # Align center
        self.preview.setFixedSize(640, 480)

        # Camera selector (IDs 0‑5 are usually enough)
        self.cam_box = QComboBox(); self.cam_box.addItems([str(i) for i in range(5)])

        # Buttons
        self.btn_start = QPushButton("Start Camera")
        self.btn_video = QPushButton("Open Video…")
        self.btn_stop  = QPushButton("Stop")

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_stop.clicked.connect(self.stop_stream);
        self.btn_stop.setEnabled(False)

        # Layout
        layout = QVBoxLayout()
        for w in (self.preview, self.cam_box, self.btn_start, self.btn_video, self.btn_stop):
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
            print("Unable to open source")
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

    # ---------- FRAME LOOP ---------- #

    def next_frame(self):
        if not (self.cap and self.cap.isOpened()):
            self.stop_stream(); return
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file:
                print("Video finished")
            self.stop_stream(); return

        # Inference (BGR→RGB) at 640‑px longer side
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _MODEL(img, size=640)  # returns list length=1
        annotated = results.render()[0]  # already BGR

        # Qt preview
        h, w, ch = annotated.shape
        qt_img = QImage(annotated.data, w, h, ch * w, QImage.Format_BGR888)
        self.preview.setPixmap(QPixmap.fromImage(qt_img))


# --------------------  MAIN  -------------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DetectorGUI(); gui.show()
    sys.exit(app.exec_())
