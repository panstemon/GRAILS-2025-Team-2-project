# Environmental Monitoring for Visually Impaired Individuals Through AI-Assisted Audio Guidance

This repository contains a project aimed at enhancing the mobility and safety of visually impaired individuals by developing an augmented-reality audio stream that monitors the surrounding environment. Leveraging cutting-edge machine-learning and computer-vision algorithms, the system provides real-time audio feedback about environmental objects and potential hazards.

---

## Overview
Visually impaired individuals face significant challenges navigating urban and indoor environments due to insufficient accessible infrastructure and the inability to visually detect obstacles. This project seeks to overcome these barriers through augmented-reality audio guidance, enhancing quality of life and autonomy.

The solution identifies and prioritises objects and obstacles **in real time**, delivering concise audio feedback for safer navigation and improved spatial awareness.

---

## Project Structure
```text
GRAILS-2025-TEAM-2-PROJECT/
├── app/
│   ├── sounds/                 # UI & ambience audio assets
│   └── main.py                 # PyQt5 GUI + YOLO + Face-ID + audio feedback
│
├── datasets/                   # Raw datasets (compressed)
│   ├── .gitkeep
│   ├── bicycle-image-dataset-vehicle-dataset.zip
│   ├── caltech101-stop-sign-images-annotations.zip
│   ├── carla-traffic-lights-images.zip
│   ├── knife-dataset.zip
│   ├── multiclassimagedatasetairplanecar.zip
│   ├── spoonvsfork.zip
│   ├── train-images-dataset.zip
│   └── vehicle-type-recognition.zip
│
├── metrics/
│   ├── evaluation_results/     # Pickled evaluation runs
│   ├── yolov5/                 # YOLOv5 cloned source (for local hub.load)
│   ├── coco.yaml               # Dataset description for validation
│   └── evaluate.py             # CLI script to benchmark YOLO weights
│
├── models/                     # Pre-trained & custom weights
│   ├── my_model.keras          # Keras classifier for known-person faces
│   ├── yolov5nu.pt             # Custom YOLOv5-N weights
│   ├── yolov5xu.pt             # Custom YOLOv5-X weights
│   ├── yolov8n.pt              # YOLOv8-N weights
│   ├── yolov8x.pt              # YOLOv8-X weights
│   ├── yolov11n.pt             # Experimental YOLOv11-N weights
│   └── yolov11x.pt             # Experimental YOLOv11-X weights
│
├── sounds/
│   └── transport.wav           # Looping vehicle ambience (auto-generated if missing)
│
├── train/
│   └── train.py                # Script to train the face-classifier CNN
│
├── .gitattributes              # LFS tracking for large files
├── .gitignore                  # Standard ignores
├── LICENSE
└── README.md                   # ← you are here
```

| Folder        | Purpose                                                                           |
| ------------- | --------------------------------------------------------------------------------- |
| **app/**      | End-user application: PyQt5 interface, detection loop, TTS & audio logic.         |
| **datasets/** | Compressed datasets referenced by training and evaluation scripts.                |
| **metrics/**  | Utilities and outputs for quantitative model evaluation (mAP, precision, recall). |
| **models/**   | All YOLO and Keras weights used by the application.                               |
| **sounds/**   | Audio assets (generated or custom).                                               |
| **train/**    | Training code for the face-recognition CNN.                                       |

---

## Team

* **Oleksandr Mazurets** – Faculty Advisor
* **Olena Sobko** – Team Captain, Researcher
* **Rostyslav Dydo** – ML Engineer, Software Developer
* **Bohdan Denysenko** – Speaker, Software Developer
* **Daryna Hardysh** – Researcher, Dataset Assembler
* **Ruslan Poplavsky** – Researcher, Dataset Assembler

---

## Datasets

This project relies on several publicly available datasets (all stored as ZIP archives inside [`datasets/`](datasets/)):

| Purpose               | Dataset / Link                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Bicycle               | [Bicycle Image Dataset](https://www.kaggle.com/datasets/dataclusterlabs/bicycle-image-dataset-vehicle-dataset)      |
| Airplane / Car / Ship | [Multiclass Image Dataset](https://www.kaggle.com/datasets/abtabm/multiclassimagedatasetairplanecar)                |
| Vehicle types         | [Vehicle Type Recognition](https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition)                   |
| Traffic lights        | [Carla Traffic Lights Images](https://www.kaggle.com/datasets/sachsene/carla-traffic-lights-images)                 |
| Stop signs            | [Caltech101 Stop Sign Images](https://www.kaggle.com/datasets/maricinnamon/caltech101-stop-sign-images-annotations) |
| Train                 | [Train Images Dataset](https://www.kaggle.com/datasets/nandepuvamsi/train-images-dataset)                           |
| Fork / Spoon          | [Spoon vs Fork](https://www.kaggle.com/datasets/kilianovski/spoonvsfork)                                            |
| Knife                 | [Knife Dataset](https://www.kaggle.com/datasets/shank885/knife-dataset)                                             |

We sincerely appreciate the efforts of all authors and organisations who provided these datasets.

---

## Installation

> The instructions below assume **Windows 10 / 11** or **Ubuntu 22.04+** with a recent NVIDIA GPU.
> CPU-only setups also work—just omit the CUDA-specific steps.

1. **Clone the repository (with Git LFS enabled)**

   ```bash
   git lfs install            # one-time setup
   git clone https://github.com/your-org/grails-2025-team-2-project.git
   cd grails-2025-team-2-project
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install PyTorch**
   *Pick the command that matches your CUDA version (or CPU-only).*
   Example for CUDA 12.1:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For CPU-only:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install project dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, manually run:

   ```bash
   pip install ultralytics==8.1.0 pyqt5 opencv-python numpy matplotlib scikit-learn seaborn face_recognition pyttsx3
   ```

   **Windows only:**

   * Face-recognition’s dlib wheels require the **MSVC ++ Build Tools**.
   * If installation fails, download the pre-built wheel from [PyPI](https://pypi.org/project/face-recognition/#files) or [this unofficial repo](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib) and install with `pip install <wheel.whl>`.

5. **(Optional) Download additional YOLO weights**
   The repo already ships with several `.pt` files. To fetch more, run:

   ```bash
   yolo export model=yolov5s.pt format=pt
   ```

6. **Verify the installation**

   ```bash
   python - << "PY"
   import torch, cv2, PyQt5
   print("CUDA:", torch.cuda.is_available())
   PY
   ```

---

## How to Run

### 1. Graphical Application

```bash
python app/main.py
```

* **Start Camera** – select a webcam index and begin live detection.
* **Open Video …** – process an existing video file (`.mp4`, `.avi`, `.mkv`, …).
* **Add Person …** – import one or more face images and label them; the model will speak their names the next time they appear.
* A single looping **transport.wav** ambience is automatically generated if missing. Its volume scales with the largest vehicle’s bounding-box area.

### 2. YOLO Evaluation CLI

Benchmark any model against a dataset:

```bash
python metrics/evaluate.py --version 8n --dataset metrics/coco.yaml
# results → metrics/evaluation_results/yolo8n_evaluation_results.pkl
```

To analyse a previous run:

```bash
python metrics/evaluate.py --analyze metrics/evaluation_results/yolo8n_evaluation_results.pkl
```

### 3. Training the Face-Classifier CNN

```bash
cd train
python train.py
```

Place your class folders under `datasets/persons/` (one folder = one person).
The script trains a 32×32 CNN and saves the model to **models/my\_model.keras**, then writes class names to `datasets/persons/class_names.txt` so the GUI can load them.

---

## License

*(To be finalised after internal review. Until then, all code is provided for non-commercial research and educational use.)*

---

### Trouble-shooting

| Issue                                       | Fix                                                                                            |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **`dlib` build errors on Windows**          | Install the pre-built wheel or Visual C++ Build Tools.                                         |
| **Black screen / “Video finished” message** | Ensure the video path is correct and supported by OpenCV FFmpeg build.                         |
| **No audio output**                         | Check that your OS default playback device is active; PyQt5 uses Qt Multimedia’s default sink. |
| **CUDA OOM**                                | Run on CPU (`set CUDA_VISIBLE_DEVICES=`) or reduce `imgsz` in `app/main.py`.                   |

---

*Maintained with ♥ by the GRAILS-2025 Team 2.*