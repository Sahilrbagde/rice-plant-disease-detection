from ultralytics import YOLO
import cv2
import os
import numpy as np

# ==========================================
# PATHS
# ==========================================

BASE_DIR = r"D:\College\internship\COE\plant traning"

MODEL_PATH = os.path.join(BASE_DIR, "best_ncnn_model")
IMAGE_FOLDER = os.path.join(BASE_DIR, "Tungro")
SAVE_FOLDER = os.path.join(BASE_DIR, "Results")

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH, task="detect")

print("Model Loaded Successfully!\n")

# ==========================================
# IMAGE EXTENSIONS
# ==========================================

extensions = (".jpg", ".jpeg", ".png", ".bmp")

images = [f for f in os.listdir(IMAGE_FOLDER)
          if f.lower().endswith(extensions)]

print(f"Found {len(images)} images.\n")

# ==========================================
# PROCESS IMAGES
# ==========================================

for img_name in images:

    print("="*60)
    print("Processing :", img_name)

    image_path = os.path.join(IMAGE_FOLDER, img_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot open image.")
        continue

    results = model(image, conf=0.5, verbose=False)

    original = image.copy()
    detected = results[0].plot()

    print("\nDetection Result")

    if len(results[0].boxes) == 0:
        print("No Disease Detected")

    for box in results[0].boxes:

        cls = int(box.cls.item())
        conf = float(box.conf.item())

        print(f"Class      : {model.names[cls]}")
        print(f"Confidence : {conf:.2f}")

    DISPLAY_HEIGHT = 600

    w1 = int(original.shape[1] * DISPLAY_HEIGHT / original.shape[0])
    w2 = int(detected.shape[1] * DISPLAY_HEIGHT / detected.shape[0])

    original = cv2.resize(original, (w1, DISPLAY_HEIGHT))
    detected = cv2.resize(detected, (w2, DISPLAY_HEIGHT))

    cv2.putText(original,
                "Original",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(detected,
                "Detection",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    BORDER = 8

    original = cv2.copyMakeBorder(
        original,
        BORDER,BORDER,BORDER,BORDER,
        cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )

    detected = cv2.copyMakeBorder(
        detected,
        BORDER,BORDER,BORDER,BORDER,
        cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )

    gap = np.full(
        (original.shape[0],25,3),
        255,
        dtype=np.uint8
    )

    combined = cv2.hconcat([original,gap,detected])

    save_path = os.path.join(SAVE_FOLDER, img_name)

    cv2.imwrite(save_path, combined)

    print("Saved :", save_path)

print("\n")
print("="*60)
print("Finished Processing All Images!")
print("="*60)