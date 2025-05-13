# -*- coding: utf-8 -*-
"""
YOLOv5 + Face-Recognition + Audio Feedback GUI (PyQt5)
──────────────────────────────────────────────────────
• Adds *one* looping transport sound (transport.wav) whose volume scales
  with the apparent size of any vehicle (car, bus, truck, etc.).
  − Silent while the bbox area is tiny (<< frame area).
• Known faces are announced at most once every **12 s** – far less chatty.
• If transport.wav is missing, a 440 Hz sine-tone WAV is auto-generated.

Install (extra deps):
    pip install pyttsx3
Run:
    python yolo_face_audio_gui.py
"""

import os, sys, pickle, time, threading, warnings, wave, struct, math
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import pyttsx3
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QComboBox, QVBoxLayout, QWidget, QInputDialog, QTextEdit,
)
from PyQt5.QtMultimedia import QSoundEffect

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────  Torch & OpenCV tweaks  ─────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
cv2.ocl.setUseOpenCL(False)
try:
    import torch._dynamo as _dynamo; _dynamo.disable()
except Exception:
    pass

torch.set_grad_enabled(False)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────  Face DB helpers  ─────────────
try:
    import face_recognition              # type: ignore
    _FACE_OK = True
except ImportError:
    print("face_recognition not installed – face ID disabled")
    _FACE_OK = False

_FACE_DB = Path("known_faces.pkl")


def _load_faces() -> Tuple[List[str], List[np.ndarray]]:
    if _FACE_DB.exists():
        with _FACE_DB.open("rb") as f:
            data = pickle.load(f)
            return data["names"], data["encodings"]
    return [], []


def _save_faces(names: List[str], encs: List[np.ndarray]):
    with _FACE_DB.open("wb") as f:
        pickle.dump({"names": names, "encodings": encs}, f)


KNOWN_NAMES, KNOWN_ENCS = _load_faces() if _FACE_OK else ([], [])

# ─────────────  YOLO initialisation  ─────────────
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
    "person",
    # vehicles
    "car", "motorcycle", "truck", "bus", "train", "bicycle", "airplane",
    # misc
    "knife", "scissors", "baseball bat", "fork", "baseball glove",
    "tennis racket", "traffic light", "stop sign",
}
ALLOWED_IDX = {i for i, n in _MODEL.names.items() if n in ALLOWED_CLASSES}

# Deterministic BGR colour per class id
_COLOURS: Dict[int, Tuple[int, int, int]] = {}


def _colour(cls_id: int):
    if cls_id not in _COLOURS:
        rng = np.random.RandomState(cls_id * 13 + 17)
        _COLOURS[cls_id] = tuple(int(x) for x in rng.randint(50, 256, 3))
    return _COLOURS[cls_id]


# ─────────────  Audio helpers  ─────────────
SOUND_DIR = Path("sounds")
SOUND_DIR.mkdir(exist_ok=True)
TRANSPORT_WAV = SOUND_DIR / "transport.wav"
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "truck",
    "bus",
    "train",
    "bicycle",
    "airplane",
}

# Generate a 1-s 440 Hz tone if wav is missing
def _ensure_transport_wav():
    if TRANSPORT_WAV.exists():
        return
    framerate = 44100
    duration = 1.0
    amp = 16000
    # convert Path to str and open in write-binary mode
    with wave.open(str(TRANSPORT_WAV), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        for i in range(int(duration * framerate)):
            val = int(amp * math.sin(2 * math.pi * 440 * i / framerate))
            wf.writeframes(struct.pack("<h", val))


_ensure_transport_wav()


def _load_sound(path: Path) -> QSoundEffect:
    se = QSoundEffect()
    se.setSource(QUrl.fromLocalFile(str(path.resolve())))
    se.setLoopCount(QSoundEffect.Infinite)
    se.setVolume(0.0)
    se.play()
    return se


class AmbientSoundManager:
    """Single looping sound whose volume reflects nearest vehicle size."""
    MAX_VOL = 0.4        # 40 %
    MIN_REL_AREA = 0.008  # ignore blobs that cover <0.8 % of frame
    SCALE = 5.0          # relat.area * SCALE → volume curve

    def __init__(self):
        self.effect = _load_sound(TRANSPORT_WAV)

    def update(self, frame_area: float, detections: List[Tuple[str, float]]):
        biggest = 0.0
        for cls, area in detections:
            if cls in VEHICLE_CLASSES:
                biggest = max(biggest, area)
        rel = biggest / frame_area
        if rel < self.MIN_REL_AREA:
            vol = 0.0
        else:
            vol = min(rel * self.SCALE, 1.0) * self.MAX_VOL
        self.effect.setVolume(vol)


# ─────────────  Text-to-speech  ─────────────
class VoiceAnnouncer:
    GAP = 12.0  # seconds between same (name, dir)

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self._queue: List[str] = []
        self._last: Dict[str, float] = {}
        th = threading.Thread(target=self._run, daemon=True)
        th.start()

    def say(self, key: str, text: str):
        now = time.time()
        if now - self._last.get(key, 0.0) >= self.GAP:
            self._last[key] = now
            self._queue.append(text)

    def _run(self):
        while True:
            if self._queue:
                try:
                    self.engine.say(self._queue.pop(0))
                    self.engine.runAndWait()
                except RuntimeError:
                    pass
            time.sleep(0.1)


# ─────────────  Face-ID helper  ─────────────
class _FaceID:
    EVERY_N = 4
    DOWNSCALE = 0.75
    TOLERANCE = 0.48

    def __init__(self):
        self._frame_id = 0
        self._cache: dict[Tuple[int, int, int, int], str] = {}
        self._encs = (
            np.vstack(KNOWN_ENCS) if KNOWN_ENCS else np.empty((0, 128))
        )

    def name_for(self, rgb_roi: np.ndarray, bbox) -> str | None:
        self._frame_id += 1
        bx1, by1, bx2, by2 = bbox
        # cache hit?
        for (cx1, cy1, cx2, cy2), cname in list(self._cache.items()):
            inter = max(0, min(bx2, cx2) - max(bx1, cx1)) * max(
                0, min(by2, cy2) - max(by1, cy1)
            )
            area1 = (bx2 - bx1) * (by2 - by1)
            area2 = (cx2 - cx1) * (cy2 - cy1)
            if inter / (area1 + area2 - inter + 1e-6) > 0.4:
                return cname
        if self._frame_id % self.EVERY_N:
            return None
        if self.DOWNSCALE != 1.0:
            rgb_roi = cv2.resize(
                rgb_roi, dsize=None, fx=self.DOWNSCALE, fy=self.DOWNSCALE
            )
        encs = face_recognition.face_encodings(rgb_roi, num_jitters=1)
        if not (encs and self._encs.size):
            return None
        dists = face_recognition.face_distance(self._encs, encs[0])
        best_idx = np.argmin(dists)
        if dists[best_idx] < self.TOLERANCE:
            name = KNOWN_NAMES[best_idx]
            self._cache[bbox] = name
            return name
        return None


_FACE_ID = _FaceID()

# ─────────────  GUI  ─────────────
class DetectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 + Face-ID + Audio")
        self.setGeometry(100, 100, 840, 840)

        self.preview = QLabel(alignment=Qt.AlignCenter)
        self.preview.setFixedSize(640, 480)

        self.log = QTextEdit(readOnly=True)
        self.log.setFixedHeight(140)
        self.log.setStyleSheet("font-family: Consolas;")

        self.cam_sel = QComboBox()
        self.cam_sel.addItems([str(i) for i in range(5)])
        self.btn_start = QPushButton("Start Camera")
        self.btn_video = QPushButton("Open Video …")
        self.btn_stop = QPushButton("Stop")
        self.btn_face = QPushButton("Add Person …")

        self.btn_face.setEnabled(_FACE_OK)
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._start_cam)
        self.btn_video.clicked.connect(self._open_video)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_face.clicked.connect(self._add_person)

        lay = QVBoxLayout()
        for w in (
            self.preview,
            self.log,
            self.cam_sel,
            self.btn_start,
            self.btn_video,
            self.btn_stop,
            self.btn_face,
        ):
            lay.addWidget(w)
        c = QWidget()
        c.setLayout(lay)
        self.setCentralWidget(c)

        self.cap = None
        self.file_mode = False
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._next)

        self.sound = AmbientSoundManager()
        self.voice = VoiceAnnouncer()

    # helpers ----------------------------------------------------------
    def _log(self, msg: str):
        self.log.append(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # Source mgmt ------------------------------------------------------
    def _start_cam(self):
        self._start(cv2.VideoCapture(int(self.cam_sel.currentText())), False)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self._start(cv2.VideoCapture(path), True)

    def _start(self, cap: cv2.VideoCapture, is_file: bool):
        self._stop()
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Unable to open source")
            return
        self.cap, self.file_mode = cap, is_file
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.log.clear()
        for b in (self.btn_start, self.btn_video):
            b.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.timer.start()

    def _stop(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.preview.clear()
            for b in (self.btn_start, self.btn_video):
                b.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.sound.update(1.0, [])  # mute
            if _DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Face DB ----------------------------------------------------------
    def _add_person(self):
        if not _FACE_OK:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select face images", "", "Images (*.jpg *.png *.jpeg)"
        )
        if not files:
            return
        name, ok = QInputDialog.getText(self, "Person Name", "Enter name:")
        name = name.strip()
        if not ok or not name:
            return
        new_encs: List[np.ndarray] = []
        for f in files:
            img = face_recognition.load_image_file(f)
            locs = face_recognition.face_locations(img, model="hog")
            if locs:
                new_encs.append(face_recognition.face_encodings(img, locs)[0])
        if not new_encs:
            QMessageBox.information(self, "Face Add", "No faces found.")
            return
        KNOWN_NAMES.extend([name] * len(new_encs))
        KNOWN_ENCS.extend(new_encs)
        _save_faces(KNOWN_NAMES, KNOWN_ENCS)
        QMessageBox.information(
            self, "Face Add", f"Added {len(new_encs)} image(s) for {name}."
        )

    # Main loop --------------------------------------------------------
    def _next(self):
        if not (self.cap and self.cap.isOpened()):
            self._stop()
            return
        ok, frame = self.cap.read()
        if not ok:
            if self.file_mode:
                print("Video finished")
            self._stop()
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = _MODEL(img_rgb, size=640).pred[0]

        if pred is not None and pred.shape[0]:
            keep = torch.tensor(
                [int(c) in ALLOWED_IDX for c in pred[:, 5]],
                device=pred.device,
                dtype=torch.bool,
            )
            pred = pred[keep]
        else:
            pred = torch.empty((0, 6))

        boxes = pred.cpu().numpy()
        labels = [_MODEL.names[int(cls)] for *_, cls in boxes]

        detections_audio: List[Tuple[str, float]] = []
        h, w, _ = frame.shape

        # Face recognition & TTS
        if _FACE_OK and len(boxes):
            dirs = {0: "left", 1: "ahead", 2: "right"}
            for idx, (*xyxy, conf, cls) in enumerate(boxes):
                if int(cls) != 0:
                    continue  # only person
                x1, y1, x2, y2 = map(int, xyxy)
                pad = int(0.1 * (y2 - y1))
                roi = img_rgb[
                    max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad
                ]
                if roi.size < 1024:
                    continue
                name = _FACE_ID.name_for(roi, (x1, y1, x2, y2))
                if name:
                    labels[idx] = name
                    dir_idx = min(int(((x1 + x2) / 2) / w * 3), 2)
                    key = f"{name}-{dir_idx}"
                    self.voice.say(key, f"{name} on your {dirs[dir_idx]}")

        # Audio: collect vehicle areas
        for (*xyxy, conf, cls), lbl in zip(boxes, labels):
            x1, y1, x2, y2 = xyxy
            area = (x2 - x1) * (y2 - y1)
            detections_audio.append((lbl, area))

        self.sound.update(w * h, detections_audio)

        # Draw + log
        if len(boxes):
            for (*xyxy, conf, cls), lbl in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _colour(int(cls)), 2)
                (tw, th), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), _colour(int(cls)), -1
                )
                cv2.putText(
                    frame,
                    lbl,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                )
                self._log(f"{lbl} – {conf:.2f}")
        else:
            self._log("–")

        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.preview.setPixmap(QPixmap.fromImage(qimg))


# ─────────────  Main  ─────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DetectorGUI()
    gui.show()
    sys.exit(app.exec())
