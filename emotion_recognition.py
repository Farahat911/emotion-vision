import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
from PIL import Image
from torchvision import models, transforms

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
GOLD = (0, 215, 255)          # BGR
FONT = cv2.FONT_HERSHEY_DUPLEX

# ─── Model ────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("best_emotion_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ─── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── MediaPipe Face Detection ────────────────────────────────────────────────
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def draw_viewfinder(frame, x1, y1, x2, y2, color=GOLD, length=25, thickness=3):
    """Draw viewfinder-style corner brackets."""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

def draw_text_with_background(frame, text, x, y, color=GOLD):
    """Render text with black semi-transparent background."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    pad = 6
    # Background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # Text on top
    cv2.putText(frame, text, (x, y), FONT, 0.8, color, 2)

def predict_emotion(face_rgb):
    """Run inference on a single face crop.  Returns emotion label string."""
    with torch.no_grad():
        pil = Image.fromarray(face_rgb)
        tensor = transform(pil).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
        return EMOTIONS[pred]

# ─── Main Loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit(1)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)

            # Skip degenerate crops
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            # Crop face region from the original BGR frame and convert to RGB
            face_bgr = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            # Predict emotion (with safety net)
            try:
                label = predict_emotion(face_rgb)
            except Exception as e:
                label = "Error"

            # Draw UI
            draw_viewfinder(frame, x1, y1, x2, y2)
            draw_text_with_background(frame, label, x1, y1 - 10)

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
