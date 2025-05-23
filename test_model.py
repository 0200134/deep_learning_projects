import torch
import cv2
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from main import DeepFakeDetector  # Import trained model

# ==============================#
#  Configuration & Validations
# ==============================#
test_folder = "./test_image/"
model_path = "deepfake_detector.pth"

# Validate test image folder
if not os.path.exists(test_folder):
    print("❌ `test_image/` folder does not exist. Add images and retry.")
    exit()

# Retrieve test images
test_images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]
if not test_images:
    print("❌ No images found in `test_image/`. Add images and try again.")
    exit()

print(f"✅ {len(test_images)} test images found. Starting detection...")

# ==============================#
#  Load DeepFake Detector Model
# ==============================#
model = DeepFakeDetector()
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Successfully loaded model!")
except RuntimeError as e:
    print(f"❌ Model loading error: {e}")
    exit()

# ==============================#
#  Define Image Preprocessing
# ==============================#
transform = transforms.Compose([
    transforms.Resize((16, 8)),  # Resized for correct 128 input features
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust normalization
])

# ==============================#
#  Image Comparison Functions
# ==============================#
def compare_images(img1, img2, threshold=10):
    """Compares pixel differences between two images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    difference = np.abs(img1 - img2)
    similar_pixels = np.where(difference < threshold, 1, 0)
    num_different_pixels = np.sum(similar_pixels == 0)

    return num_different_pixels

def compute_similarity(img1, img2):
    """Computes Structural Similarity Index (SSIM) between images."""
    return ssim(img1, img2, data_range=img1.max() - img1.min())

# ==============================#
#  Process Test Images
# ==============================#
previous_image_tensor = None  # Store previous image for comparison

for image_file in test_images:
    image_path = os.path.join(test_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error loading {image_path}. Skipping...")
        continue

    # Convert OpenCV image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply transformations
    image_tensor = transform(image_pil).unsqueeze(0)

    # Fix shape issue - ensuring correct input dimensions for model
    image_tensor = image_tensor.flatten().unsqueeze(0)  # Flatten before inference
    print(f"✅ Fixed input tensor shape: {image_tensor.shape}")  # Debugging step

    # Run deepfake detection
    with torch.no_grad():
        output = model(image_tensor)
        prediction = "Fake" if output.item() > 0.5 else "Real"

    print(f"🔍 {image_file}: Prediction -> {prediction}")

    # Pixel-based and SSIM comparisons (if a previous image exists)
    if previous_image_tensor is not None:
        prev_image = previous_image_tensor.squeeze().numpy()
        current_image = image_tensor.squeeze().numpy()

        pixel_differences = compare_images(prev_image, current_image)
        similarity_score = compute_similarity(prev_image, current_image)

        print(f"🧐 Pixel Differences: {pixel_differences} | SSIM Score: {similarity_score:.4f}")

    # Store current image for next comparison
    previous_image_tensor = image_tensor

print("✅ All test images processed successfully.")
