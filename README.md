# 🎭 Video Emotion Analytics Pipeline

A real-time facial emotion recognition system built from scratch using PyTorch, MediaPipe, and OpenCV. Supports both live webcam mode and full video file analysis with CSV reporting.

---

## 🚀 Features

- **Real-time Mode** — Live webcam feed with instant emotion detection
- **Video Analysis Mode** — Process any `.mp4` file and export a detailed emotion report
- **7 Emotion Classes** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Dynamic UI** — Color-coded viewfinder that changes per emotion
- **CSV Export** — Frame-by-frame emotion log for every detected face
- **Optimized Performance** — Frame skipping + batch inference to reduce GPU load

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch + ResNet18 | Emotion classification model |
| MediaPipe | Fast face detection |
| OpenCV | Video I/O and UI rendering |
| Pillow | Image preprocessing |
| NumPy | Array operations |

---

## 📁 Project Structure

```
ai/
├── best_emotion_model.pth       # Trained ResNet18 weights
├── emotion_recognition.py       # Real-time webcam mode
├── video_emotion_pipeline.py    # Video file analysis mode
├── input.mp4                    # Input video (place your video here)
├── output_analyzed.mp4          # Output video with emotion overlays
├── emotion_report.csv           # Generated emotion report
└── requirements.txt
```

---

## ⚙️ Installation

**Requirements:** Python 3.11

```bash
# Create virtual environment
py -3.11 -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Real-time Webcam Mode
```bash
python emotion_recognition.py
```
Press `q` to quit.

### Video File Analysis Mode
1. Place your video file in the project folder and rename it `input.mp4`
2. Run:
```bash
python video_emotion_pipeline.py
```
3. Output:
   - `output_analyzed.mp4` — Video with emotion overlays
   - `emotion_report.csv` — Full frame-by-frame emotion log

---

## 📊 CSV Report Format

| Column | Description |
|--------|-------------|
| Frame | Frame index |
| Emotion | Detected emotion label |
| Face_Index | Face ID within that frame |

---

## 🧠 Engineering Highlights

### 1. Temporal Smoothing (Prediction Cooldown)
Inference runs every N frames instead of every frame, eliminating flickering and cutting GPU usage significantly. Labels from the last inference are reused in between.

### 2. Smart Frame Sync
GPU processes frames faster than the original video FPS, which would cause fast-forward playback. The pipeline calculates elapsed time per frame and applies a dynamic delay to maintain cinematic speed.

### 3. Batch Inference
When multiple faces appear in the same frame, they're processed in a single batched forward pass instead of sequential calls — faster and more memory-efficient.

### 4. Dynamic Color UI
Each emotion maps to a unique BGR color. The viewfinder brackets and label change color in real-time based on the detected emotion.

---

## 📈 Emotion Color Map

| Emotion | Color |
|---------|-------|
| Angry | 🔴 Red |
| Disgust | 🟢 Green |
| Fear | 🟣 Purple |
| Happy | 🟡 Gold |
| Sad | 🔵 Blue |
| Surprise | 🩵 Cyan |
| Neutral | ⚪ White |

---

## 📌 Notes

- Model trained on FER2013 dataset
- Optimized for CPU — GPU accelerates if available (CUDA)
- MediaPipe version must be `0.10.9` for `mp.solutions` compatibility

---

## 👤 Author

**Mohammed Farahat**  
2nd Year AI Engineering Student — Mansoura National University  
[LinkedIn](#) | [GitHub](#)
