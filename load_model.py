from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import sys

# ==============================#
#     Load DeepFake Model
# ==============================#
class DeepFakeDetector:
    def __init__(self):
        try:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)  # Binary classification
            self.model.load_state_dict(torch.load("deepfake_detector.pth", map_location=torch.device('cpu')), strict=False)
            self.model.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def predict(self, image_path):
        if not self.model:
            return "Error: Model not loaded."

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return "Error: Unable to load image."

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(img).unsqueeze(0)  # Fix tensor shape

        with torch.no_grad():
            prediction = self.model(img_tensor).item()
            return "Fake" if prediction > 0.5 else "Real"

# ==============================#
#     GUI Application
# ==============================#
class DeepFakeApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Fix: Adjust window settings properly
        self.setWindowTitle("Deepfake Detection")
        self.setGeometry(100, 100, 500, 600)  
        self.setMinimumSize(500, 600)  # Prevent scaling issues

        self.detector = DeepFakeDetector()

        # UI Elements
        layout = QVBoxLayout()
        self.label = QLabel("이미지를 업로드하세요.", self)
        layout.addWidget(self.label)

        self.image_label = QLabel(self)  # Label for detected images
        layout.addWidget(self.image_label)

        self.button = QPushButton("이미지 선택", self)
        self.button.clicked.connect(self.detect_deepfake)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def detect_deepfake(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg)")
        if not image_path:
            self.label.setText("이미지를 선택하세요.")
            return

        result = self.detector.predict(image_path)
        if "Error" in result:
            self.label.setText(result)
            return

        # Fix: Ensure image loads properly
        img = cv2.imread(image_path)
        if img is None:
            self.label.setText("❌ 이미지 로딩 실패.")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            color = (0, 0, 255) if result == "Fake" else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Fix: Ensure GUI updates properly
        processed_image_path = "processed_image.jpg"
        cv2.imwrite(processed_image_path, img)

        self.image_label.setPixmap(QPixmap(processed_image_path))
        self.label.setText(f"결과: {result}")

# ==============================#
#     Run Application
# ==============================#
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepFakeApp()
    window.show()
    sys.exit(app.exec())
