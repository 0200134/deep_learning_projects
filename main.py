import os
import cv2
import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# ==============================#
#         Configuration
# ==============================#
DATASET_PATH = "./images/"
OUTPUT_DIR = "./processed_frames/"
MODEL_SAVE_PATH = "./deepfake_detector.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================#
#    Load Fortran Libraries
# ==============================#
try:
    lib_features = ctypes.CDLL("./libfeatures.dll")
except Exception as e:
    print(f"❌ Error loading feature extraction DLL: {e}")
    exit()

# ==============================#
#   Feature Extraction Function
# ==============================#
def extract_features(image_path):
    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print(f"⚠ Skipping unreadable image: {image_path}")
        return None

    try:
        resized = cv2.resize(frame, (128, 128))
    except Exception as e:
        print(f"❌ Resize error: {e}")
        return None

    frame_fortran = np.asfortranarray(resized, dtype=np.float32)
    features = np.zeros(128, dtype=np.float32, order="F")

    try:
        lib_features.extract_features(
            frame_fortran.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

    return features if np.any(features) else None

# ==============================#
#     Define Model
# ==============================#
class DeepFakeDetector(nn.Module):
    def __init__(self, input_size=128):
        super(DeepFakeDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ==============================#
#   Load or Train Model
# ==============================#
model = DeepFakeDetector(input_size=128)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()

if os.path.exists(MODEL_SAVE_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"✅ Loaded model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ Model loading error: {e}")

# ==============================#
#     Train Model
# ==============================#
feature_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")]
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for feat_file in feature_files:
        features = np.load(os.path.join(OUTPUT_DIR, feat_file))
        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0)
        label = torch.tensor([[1.0 if "fake" in feat_file.lower() else 0.0]], dtype=torch.float)

        optimizer.zero_grad()
        output = model(features_tensor)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.6f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# ==============================#
#  Real-Time Deepfake Detection
# ==============================#
def detect_deepfake_live():
    cap = cv2.VideoCapture(0)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            resized = cv2.resize(gray, (128, 128))
        except Exception as e:
            print(f"❌ Resize error in live detection: {e}")
            continue

        frame_fortran = np.asfortranarray(resized, dtype=np.float32)
        features = np.zeros(128, dtype=np.float32, order="F")

        try:
            lib_features.extract_features(
                frame_fortran.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        except Exception as e:
            print(f"❌ Feature extraction error in live detection: {e}")
            continue

        if not np.any(features):
            continue

        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            prediction = model(features_tensor).item()
            label_str = "Fake" if prediction > 0.5 else "Real"

        cv2.putText(frame, f"Prediction: {label_str}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("🎥 To run live detection, call detect_deepfake_live()")

# Uncomment to run live detection
# detect_deepfake_live()
