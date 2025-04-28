# -*- coding: utf-8 -*-
"""
YOLOv5 + Face‑Recognition + Smart‑Audio GUI         (PyQt5)
───────────────────────────────────────────────────────────
• Real‑time object detection with Ultralytics YOLOv5
• Optional face recognition with persistent database
• SINGLE, prioritised, non‑spammy voice call‑outs
      – “Danger” items (knife, bat …) top priority
      – Known people next, vehicles last
      – Max one sentence per second, with 3‑s cooldown per sentence
Extra deps
──────────
$ pip install pyqt5 opencv‑python torch torchvision torchaudio face_recognition pyttsx3
"""

import sys, time, queue, pickle, threading
from pathlib import Path
from typing import List, Tuple

import cv2, numpy as np, torch, pyttsx3
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QWidget, QInputDialog, QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

# --------------------  TTS ENGINE  -------------------- #
_speech_q: "queue.Queue[str]" = queue.Queue(maxsize=1)  # drop if busy
_last_spoken: dict[str, float] = {}
_COOLDOWN = 3.0          # per‑sentence
_MIN_INTERVAL = 1.0      # across whole stream
_last_global = 0.0

def _tts_worker():
    engine = pyttsx3.init()
    while True:
        text = _speech_q.get()
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text: str):
    global _last_global
    now = time.time()
    if now - _last_global < _MIN_INTERVAL:       # overall throttle
        return
    if now - _last_spoken.get(text, 0) < _COOLDOWN:
        return
    if _speech_q.full():
        return
    _last_spoken[text] = now
    _last_global = now
    _speech_q.put(text)

# --------------------  FACE DB  ---------------------- #
try:
    import face_recognition
    _FACE_OK = True
except ImportError:
    print("face_recognition not installed – face ID disabled")
    _FACE_OK = False

_FACE_DB = Path("known_faces.pkl")
def _load_faces() -> Tuple[List[str], List[np.ndarray]]:
    if _FACE_DB.exists():
        with _FACE_DB.open("rb") as f:
            data = pickle.load(f); return data["names"], data["encodings"]
    return [], []
def _save_faces(names, encs):
    with _FACE_DB.open("wb") as f:
        pickle.dump({"names": names, "encodings": encs}, f)

KNOWN_NAMES, KNOWN_ENCODINGS = _load_faces() if _FACE_OK else ([], [])

# --------------------  YOLOv5  ----------------------- #
print("Loading YOLOv5 …")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(_DEVICE).eval()
if _DEVICE == "cuda": _MODEL.half()
print("Model ready on", _DEVICE)

ALLOWED = {
    "person","knife","scissors","baseball bat","fork","baseball glove","tennis racket",
    "car","motorcycle","truck","bus","train","bicycle",
    "airplane","traffic light","stop sign",
}
ALLOWED_IDX = {i for i,n in _MODEL.names.items() if n in ALLOWED}

DANGER = {"knife","scissors","baseball bat","fork"}
VEHICLES = {"car","motorcycle","truck","bus","train","bicycle","airplane"}

def pos_str(cx, w):
    return "left" if cx < w/3 else "right" if cx > 2*w/3 else "ahead"

# --------------------  GUI  -------------------------- #
class Detector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart YOLO/Face with Audio")
        self.setGeometry(100,100,820,700)
        self.prev = QLabel(alignment=0x84); self.prev.setFixedSize(640,480)
        self.cam_box = QComboBox(); self.cam_box.addItems([str(i) for i in range(5)])
        self.b_start=QPushButton("Start")
        self.b_vid=QPushButton("Open Video…")
        self.b_stop=QPushButton("Stop")
        self.b_add=QPushButton("Add Person…")
        self.b_add.setEnabled(_FACE_OK)
        self.b_start.clicked.connect(self.start_cam); self.b_vid.clicked.connect(self.open_vid)
        self.b_stop.clicked.connect(self.stop); self.b_add.clicked.connect(self.add_person)
        self.b_stop.setEnabled(False)
        lay=QVBoxLayout(); [lay.addWidget(w) for w in (self.prev,self.cam_box,
                self.b_start,self.b_vid,self.b_stop,self.b_add)]
        c=QWidget(); c.setLayout(lay); self.setCentralWidget(c)
        self.cap=None; self.is_file=False
        self.timer=QTimer(); self.timer.timeout.connect(self.loop)

    # ---------- sources ----------
    def start_cam(self): self._start(cv2.VideoCapture(int(self.cam_box.currentText())),False)
    def open_vid(self):
        p,_=QFileDialog.getOpenFileName(self,"Video","","Video (*.mp4 *.avi *.mkv *.mov)")
        if p: self._start(cv2.VideoCapture(p),True)
    def _start(self,cap,is_file):
        self.stop();   # reset
        if not cap.isOpened(): QMessageBox.warning(self,"Error","Cannot open"); return
        self.cap, self.is_file = cap,is_file
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.b_start.setEnabled(False); self.b_vid.setEnabled(False); self.b_stop.setEnabled(True)
        self.timer.start(30)

    def stop(self):
        if self.cap:
            self.timer.stop(); self.cap.release(); self.cap=None
            self.prev.clear()
        self.b_start.setEnabled(True); self.b_vid.setEnabled(True); self.b_stop.setEnabled(False)

    # ---------- add face ----------
    def add_person(self):
        imgs,_ = QFileDialog.getOpenFileNames(self,"Faces","","Images (*.jpg *.png *.jpeg)")
        if not imgs: return
        name,ok = QInputDialog.getText(self,"Name","Person name:")
        if not ok or not name.strip(): return
        new=[]
        for p in imgs:
            img=face_recognition.load_image_file(p)
            locs=face_recognition.face_locations(img,model="hog")
            if locs: new.append(face_recognition.face_encodings(img,locs)[0])
        if not new:
            QMessageBox.information(self,"Add","No faces"); return
        KNOWN_NAMES.extend([name]*len(new)); KNOWN_ENCODINGS.extend(new); _save_faces(KNOWN_NAMES,KNOWN_ENCODINGS)
        QMessageBox.information(self,"Add",f"Added {len(new)} images for {name}")

    # ---------- main loop ----------
    def loop(self):
        if not (self.cap and self.cap.isOpened()):
            self.stop(); return
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file: print("Video end")
            self.stop(); return

        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _MODEL(rgb,size=640)
        pred = results.pred[0]
        if pred is not None and pred.shape[0]:
            m = torch.tensor([int(c) in ALLOWED_IDX for c in pred[:,5]],
                             dtype=torch.bool,device=pred.device)
            results.pred[0] = pred[m]; pred = results.pred[0]

        annotated = results.render()[0].copy()

        # Build AUDIO candidate list (priority, area, sentence)
        candidates = []

        # objects
        for *box, conf, cls in pred.tolist() if pred is not None else []:
            cls=int(cls); label=_MODEL.names[cls]
            x1,y1,x2,y2=map(int,box); area=(x2-x1)*(y2-y1); phrase=None
            if label=="person": continue  # face section handles
            direction = pos_str((x1+x2)/2,w)
            phrase = f"{label} on {direction}"
            priority = 3 if label in DANGER else 1 if label in VEHICLES else 0
            candidates.append((priority,area,phrase))

        # faces
        if _FACE_OK and KNOWN_ENCODINGS:
            small = cv2.resize(rgb,(0,0),fx=0.5,fy=0.5)
            f_locs = face_recognition.face_locations(small,model="hog")
            f_encs = face_recognition.face_encodings(small,f_locs)
            for (t,r,b,l),enc in zip(f_locs,f_encs):
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS,enc,0.5)
                if True in matches:
                    counts={}
                    for i,m in enumerate(matches):
                        if m: counts[KNOWN_NAMES[i]] = counts.get(KNOWN_NAMES[i],0)+1
                    name=max(counts,key=counts.get)
                    # upscale coords for draw
                    scale=2; tl,br = (l*scale,t*scale),(r*scale,b*scale)
                    cv2.rectangle(annotated, tl, br, (0,255,0),2)
                    cv2.rectangle(annotated, (tl[0],br[1]-20), br, (0,255,0), cv2.FILLED)
                    cv2.putText(annotated,name,(tl[0]+2,br[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
                    cx=(tl[0]+br[0])/2; direction=pos_str(cx,w)
                    area=(br[0]-tl[0])*(br[1]-tl[1])
                    candidates.append((2,area,f"{name} on {direction}"))

        # choose best candidate to speak
        if candidates:
            phrase = max(candidates,key=lambda x:(x[0],x[1]))[2]
            speak(phrase)

        # Qt preview
        qimg = QImage(annotated.data,w,h,annotated.strides[0],QImage.Format_BGR888)
        self.prev.setPixmap(QPixmap.fromImage(qimg))

# --------------------  main -------------------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Detector(); gui.show()
    sys.exit(app.exec_())
