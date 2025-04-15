import pandas as pd
import sys
import torch
import cv2
import time
import random
import pyttsx3
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import subprocess
import json
from PyQt5.QtWidgets import QTextEdit


OBJECT_PRIORITIES = {
    "knife": 1,            
    "scissors": 2,         
    "baseball bat": 3,     
    "fork": 4,             
    "baseball glove": 5,   
    "tennis racket": 6,    
    "car": 7,             
    "motorcycle": 8,      
    "truck": 9,            
    "bus": 10,             
    "train": 11,          
    "bicycle": 12,         
    "airplane": 13,        
    "traffic light": 14,   
    "stop sign": 15,       
}


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
face_recognition_model = load_model('../models/my_model.keras')

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

CLASS_NAMES = load_class_names('class_names.txt')


engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

FRAME_SKIP = 5
COLORS = {}
last_said = {}

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"speak_greeting": True}

def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

def get_color(label):
    if label not in COLORS:
        COLORS[label] = [random.randint(0, 255) for _ in range(3)]
    return COLORS[label]


def sort_by_priority(labels):
    
    return sorted(labels, key=lambda label: OBJECT_PRIORITIES.get(label, float('inf')))


def filter_objects(detected_objects):
    return [row['name'] for index, row in detected_objects.iterrows() if row['confidence'] > 0.5]

def speak_labels(labels):
    global last_said
    current_time = time.time()
    
   
    sorted_labels = sort_by_priority(labels)
    
    to_speak = [label for label in sorted_labels if label not in last_said or current_time - last_said[label] > 2]

    if to_speak:
        description = "Objects detected: " + ", ".join(to_speak)
        engine.say(description)
        engine.runAndWait()
        for label in to_speak:
            last_said[label] = current_time

def recognize_face(face):
    # Validate face dimensions
    if face is None or face.size == 0 or len(face.shape) != 3 or face.shape[2] != 3:
        return "nameless"

    try:
        # Resize and preprocess the face
        face = cv2.resize(face, (32, 32))  # Resize to 32x32 pixels
        face = face / 255.0  # Normalize pixel values to [0, 1]
        face = face.reshape(1, 32, 32, 3)  # Add batch dimension
        predictions = face_recognition_model.predict(face)  # Model prediction
        predicted_class = predictions.argmax(axis=-1)[0]
        confidence = predictions.max()

        return CLASS_NAMES[predicted_class] if confidence > 0.5 else "nameless"
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return "nameless"



class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(150, 150, 300, 200)

        layout = QVBoxLayout()

        self.greeting_checkbox = QPushButton("Enable greeting on startup", self)
        self.greeting_checkbox.setCheckable(True)
        self.greeting_checkbox.setChecked(settings.get("speak_greeting", True))
        self.greeting_checkbox.clicked.connect(self.update_greeting_setting)
        layout.addWidget(self.greeting_checkbox)

        self.setLayout(layout)

    def update_greeting_setting(self):
        settings["speak_greeting"] = self.greeting_checkbox.isChecked()
        save_settings(settings)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Detection Assistant")
        self.setGeometry(100, 100, 500, 700)

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.start_button = QPushButton("Start Detection", self)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop Detection", self)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        self.settings_button = QPushButton("Settings", self)
        self.settings_button.setGeometry(700, 10, 80, 30)
        self.settings_button.setEnabled(True)
        self.settings_button.clicked.connect(self.open_settings)

        self.text_box = QTextEdit(self)  # Add a text box to display class and confidence
        self.text_box.setReadOnly(True)
        self.text_box.setFixedHeight(100)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.text_box)  # Add the text box to the layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.frame_counter = 0
        self.detection_active = False

        self.current_function_index = 0
        self.functions = ["Start Detection", "Stop Detection", "Settings"]
        self.setMouseTracking(True)

        settings = load_settings()
        speak_greeting = settings.get("speak_greeting", True)

        if speak_greeting:
            engine.say("Hello, I am your assistant. Tap once to hear the next function, swipe right to select it. To start object detection, press the 'Start Detection' button. To stop it, press the 'Stop Detection' button.")
            engine.runAndWait()

    def start_detection(self):
        if not self.detection_active:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.detection_active = True
            engine.say("Detection started")
            engine.runAndWait()

    def stop_detection(self):
        if self.detection_active:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.video_label.clear()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.detection_active = False
            engine.say("Detection stopped")
            engine.runAndWait()

    def open_settings(self):
        self.close()
        subprocess.run(["python", "setting_wind.py"])

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_counter += 1

        if self.frame_counter % FRAME_SKIP == 0:
            results = model(frame)  # YOLO detection
            detected_objects = results.pandas().xyxy[0]

            # Convert confidence to numeric and drop NaNs
            detected_objects['confidence'] = pd.to_numeric(detected_objects['confidence'], errors='coerce')
            detected_objects = detected_objects.dropna(subset=['confidence'])

            # Filter objects based on user-defined criteria
            labels = filter_objects(detected_objects)

            # Check if "person" is detected
            if "person" in labels:
                for index, row in detected_objects.iterrows():
                    if row['name'] == "person":
                        # Crop face from the frame
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)  # Boundary checks
                        face = frame[y1:y2, x1:x2]

                        # Validate and process the face
                        if face is not None and face.size > 0 and len(face.shape) == 3 and face.shape[2] == 3:
                            try:
                                # Resize and preprocess the face
                                face_resized = cv2.resize(face, (32, 32))  # Resize to model's input size
                                face_resized = face_resized / 255.0  # Normalize
                                face_resized = face_resized.reshape(1, 32, 32, 3)  # Add batch dimension

                                # Predict and get label and confidence
                                predictions = face_recognition_model.predict(face_resized)
                                predicted_class = predictions.argmax(axis=-1)[0]
                                confidence = predictions.max()
                                face_label = CLASS_NAMES[predicted_class] if confidence > 0.5 else "Unknown"

                                # Append results to the text box
                                self.text_box.append(f"Recognized: {face_label}, Confidence: {confidence:.2f}")

                                # Draw rectangle and label around the face
                                face_color = get_color("face")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), face_color, 2)
                                cv2.putText(frame, face_label, (x1 - 50, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
                            except Exception as e:
                                print(f"Error processing face: {e}")

            # Convert frame for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))



    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.current_function_index = (self.current_function_index + 1) % len(self.functions)
            engine.say(f"Next function: {self.functions[self.current_function_index]}")
            engine.runAndWait()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            selected_function = self.functions[self.current_function_index]
            engine.say(f"You selected: {selected_function}")
            engine.runAndWait()
            if selected_function == "Start Detection":
                self.start_detection()
            elif selected_function == "Stop Detection":
                self.stop_detection()
            elif selected_function == "Settings":  
                self.open_settings()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()  
    sys.exit(app.exec_())  
