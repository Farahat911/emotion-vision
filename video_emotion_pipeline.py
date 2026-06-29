import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import time
from PIL import Image
from collections import Counter
from torchvision import models, transforms

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
FONT = cv2.FONT_HERSHEY_DUPLEX
CONFIDENCE_THRESH = 0.6

EMOTION_COLORS = {
    'Angry':    (0, 0, 255),      # Red
    'Disgust':  (0, 255, 0),      # Green
    'Fear':     (128, 0, 128),    # Purple
    'Happy':    (0, 215, 255),    # Gold
    'Sad':      (255, 0, 0),      # Blue
    'Surprise': (255, 255, 0),    # Cyan
    'Neutral':  (255, 255, 255),  # White
}

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
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=CONFIDENCE_THRESH
)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def draw_viewfinder(frame, x1, y1, x2, y2, color, length=25, thickness=3):
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

def draw_text_with_background(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, 0.8, color, 2)

def extract_face_crops(frame, detections):
    """Return list of (x1, y1, x2, y2, face_rgb_array) tuples."""
    h, w = frame.shape[:2]
    crops = []
    for det in detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        face_bgr = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        crops.append((x1, y1, x2, y2, face_rgb))
    return crops

def batch_predict(crops):
    """Run batched inference on a list of (x1, y1, x2, y2, face_rgb) tuples.
    Returns list of emotion labels aligned with the input order."""
    if not crops:
        return []
    with torch.no_grad():
        tensors = []
        for _, _, _, _, face_rgb in crops:
            pil = Image.fromarray(face_rgb)
            tensors.append(transform(pil))
        batch = torch.stack(tensors).to(DEVICE)
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().tolist()
        return [EMOTIONS[p] for p in preds]

# ─── Video I/O ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture("input.mp4")
if not cap.isOpened():
    print("Error: Cannot open input.mp4")
    exit(1)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("output_analyzed.mp4", fourcc, fps, (orig_w, orig_h))

# ─── Data Logging & Frame Skipping ────────────────────────────────────────────
process_every_n_frames = 5
frame_counter = 0
locked_labels = []       # last-known emotion labels, reused when inference is skipped
frame_log = []           # per-frame list of emotion labels
total_frames = 0

print(f"Processing input.mp4  ({orig_w}x{orig_h} @ {fps:.2f} fps)")
print(f"Inference every {process_every_n_frames} frames — press 'q' to quit early.\n")

# ─── Main Loop ────────────────────────────────────────────────────────────────
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    frame_emotions = []

    if results.detections:
        try:
            crops = extract_face_crops(frame, results.detections)

            # Run inference only every N frames OR when the face count changes
            should_infer = (frame_counter % process_every_n_frames == 0) or \
                           (len(locked_labels) != len(crops))

            if should_infer:
                locked_labels = batch_predict(crops)

            for (x1, y1, x2, y2, _), label in zip(crops, locked_labels):
                color = EMOTION_COLORS.get(label, (255, 255, 255))
                draw_viewfinder(frame, x1, y1, x2, y2, color)
                draw_text_with_background(frame, label, x1, y1 - 10, color)
                frame_emotions.append(label)
        except Exception as e:
            print(f"Warning: inference error on frame {total_frames}: {e}")

    frame_log.append(frame_emotions)
    total_frames += 1
    frame_counter += 1

    writer.write(frame)
    cv2.imshow("Video Emotion Analytics", frame)

    elapsed_ms = (time.time() - start_time) * 1000
    target_delay = 1000 / fps
    actual_delay = max(1, int(target_delay - elapsed_ms))
    if cv2.waitKey(actual_delay) & 0xFF == ord('q'):
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
writer.release()
cv2.destroyAllWindows()

# ─── CSV Export ───────────────────────────────────────────────────────────────
import csv
with open("emotion_report.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Frame", "Emotion", "Face_Index"])
    for fidx, emotions in enumerate(frame_log):
        for eidx, emo in enumerate(emotions):
            w.writerow([fidx, emo, eidx])
print(f"\nExported emotion_report.csv ({total_frames} frames)")

# ─── Terminal Summary ─────────────────────────────────────────────────────────
all_emotions = [e for frame_ems in frame_log for e in frame_ems]
total_detections = len(all_emotions)

if total_detections > 0:
    counter = Counter(all_emotions)
    print("\n=== Emotion Distribution ===")
    for emo in EMOTIONS:
        cnt = counter.get(emo, 0)
        pct = (cnt / total_detections) * 100
        print(f"  {emo:>10s}: {cnt:>5d}  ({pct:5.2f}%)")
    print(f"\n  Total faces detected across all frames: {total_detections}")
else:
    print("\nNo faces detected in any frame.")
