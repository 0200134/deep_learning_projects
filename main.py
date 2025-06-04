<<<<<<< HEAD
import sys
import os
import glob
import cv2
import ctypes
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import unicodedata
from torchvision import models, transforms
from PIL import Image, ImageSequence, ImageFile
from torch.utils.data import DataLoader, TensorDataset
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap

# Fix: Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
=======
import os
import cv2
import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
>>>>>>> d371162 (Updated deep learning project)

# ==============================#
#         Configuration
# ==============================#
<<<<<<< HEAD
DATASET_PATHS = [
    "C:/my_cpp_python_gui/FaceForensics/images",
    "C:/my_cpp_python_gui/deep_learning_projects/images",
    "C:/my_cpp_python_gui/deep_learning_projects/test_image"
]
LABELS_FILE = "C:/my_cpp_python_gui/image_labels.csv"  # Optional CSV label file
MODEL_SAVE_PATH = "./deepfake_detector.pth"
DLL_PATH = "C:/my_cpp_python_gui/libfeatures.dll"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================#
#   Load Feature Extraction DLL
# ==============================#
if os.path.exists(DLL_PATH):
    try:
        lib_features = ctypes.CDLL(DLL_PATH)
        print(f"✅ Loaded DLL: {DLL_PATH}")
    except Exception as e:
        print(f"❌ DLL Load Error: {e}")
        sys.exit(1)
else:
    print(f"❌ Missing DLL: {DLL_PATH}")
    sys.exit(1)

# Define function arguments
lib_features.extract_features.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]

# ==============================#
#   Load Image Paths & Labels
# ==============================#
def normalize_filename(filename):
    """ Normalize Unicode filenames for compatibility """
    return unicodedata.normalize('NFKD', filename).encode('utf-8', 'ignore').decode('utf-8')

def safe_file_path(image_path):
    """ Ensure file path is correctly encoded for OS compatibility """
    return os.path.normpath(image_path)

image_paths = []
labels = []

for path in DATASET_PATHS:
    if not os.path.exists(path):
        print(f"❌ Dataset folder missing: {path}")
        continue  # Fix: Continue instead of exit
    detected_files = glob.glob(os.path.join(path, "*.*"))  # Fix: Load multiple file types
    image_paths.extend(detected_files)

# Filter out missing files
image_paths = [safe_file_path(normalize_filename(img)) for img in image_paths if os.path.exists(img)]

# Debugging: Print detected files
print(f"🔍 Total images found: {len(image_paths)}")
if not image_paths:
    print("⚠ No images detected. Please check your dataset paths.")

# Assign labels dynamically
def get_label_from_filename(filename):
    return 1 if "fake" in filename.lower() else 0  # Fake = 1, Real = 0

if os.path.exists(LABELS_FILE):  # Load labels if CSV exists
    labels_df = pd.read_csv(LABELS_FILE)
    labels_dict = dict(zip(labels_df["filename"], labels_df["label"]))
    labels = [labels_dict.get(os.path.basename(img), get_label_from_filename(img)) for img in image_paths]
else:
    labels = [get_label_from_filename(img) for img in image_paths]  # Default labeling

# ==============================#
#  Feature Extraction via DLL
# ==============================#
def read_image_unicode_safe(image_path):
    """ Read image safely, including GIF support """
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None

    try:
        img = Image.open(image_path)
        img.load()  # Fix: Pre-load truncated images

        if img.format == "GIF":
            frames = [frame.convert("L") for frame in ImageSequence.Iterator(img)]
            return [cv2.resize(np.array(frame), (128, 128)) for frame in frames]
        else:
            return cv2.resize(np.array(img.convert("L")), (128, 128))
    except Exception as e:
        print(f"❌ Error reading image: {image_path} - {e}")
        return None

def extract_features(image_data):
    if isinstance(image_data, list):  # Multiple frames in GIF
        features = [extract_single_image_features(frame) for frame in image_data]
        return np.mean(features, axis=0)  # Aggregate features
    else:
        return extract_single_image_features(image_data)

def extract_single_image_features(img):
    frame_fortran = np.asfortranarray(img, dtype=np.float32)
=======
DATASET_PATH = "./faceforensics_data/"
OUTPUT_DIR = "./processed_frames/"
MODEL_SAVE_PATH = "./deepfake_detector.pth"
LIBRARY_PATH = "C:/my_cpp_python_gui/libfeatures.dll"  # Ensure correct path

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================#
#    Load Fortran Libraries
# ==============================#
try:
    lib_features = ctypes.CDLL(LIBRARY_PATH)
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
>>>>>>> d371162 (Updated deep learning project)
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
<<<<<<< HEAD
#  Convert Images to Tensors
# ==============================#
def load_images(image_paths, labels):
    feature_list = []
    label_list = []

    for img_path, label in zip(image_paths, labels):
        features = extract_features(read_image_unicode_safe(img_path))
        if features is None:
            continue  # Skip bad images
        feature_list.append(features)
        label_list.append(label)

    return torch.tensor(np.array(feature_list), dtype=torch.float), torch.tensor(label_list, dtype=torch.float)

image_tensors, labels_tensor = load_images(image_paths, labels)
train_dataset = TensorDataset(image_tensors, labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==============================#
#     Define Model (Fully Connected)
=======
#     Define Model (ResNet)
>>>>>>> d371162 (Updated deep learning project)
# ==============================#
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
<<<<<<< HEAD
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# ==============================#
#   Train Model
# ==============================#
def train_model():
    model = DeepFakeDetector().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            output = model(images)

            # Fix: Ensure outputs are between 0 and 1
            output = torch.clamp(output, min=0, max=1)
            output = torch.nan_to_num(output, nan=0.5)  # Fix NaN values

            # Debugging: Print sample outputs
            print("Sample model outputs:", output[:5].detach().cpu().numpy())

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model trained & saved to {MODEL_SAVE_PATH}")

train_model()
print(f"✅ Loaded model from {MODEL_SAVE_PATH}")
=======
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# ==============================#
#   Load or Train Model
# ==============================#
model = DeepFakeDetector()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()

if os.path.exists(MODEL_SAVE_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')), strict=False)
        model.eval()
        print(f"✅ Loaded model from {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ Model loading error: {e}")

# ==============================#
#     Train Model with FaceForensics++
# ==============================#
feature_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")]
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for feat_file in feature_files:
        features = np.load(os.path.join(OUTPUT_DIR, feat_file))
        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0).view(1, 3, 128, 128)  # Reshaped input
        label = torch.tensor([[1.0 if "fake" in feat_file.lower() else 0.0]], dtype=torch.float)

        optimizer.zero_grad()
        output = model(features_tensor)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.6f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model trained & saved to {MODEL_SAVE_PATH}")

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

        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0).view(1, 3, 128, 128)

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
>>>>>>> d371162 (Updated deep learning project)
