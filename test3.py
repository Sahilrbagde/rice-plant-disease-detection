from ultralytics import YOLO
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# ============================================
# PROJECT PATH
# ============================================

BASE_DIR = r"D:\College\internship\COE\plant traning"

MODEL_PATH = os.path.join(BASE_DIR, "best_ncnn_model")
RESULT_FOLDER = os.path.join(BASE_DIR, "Results")

os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================
# LOAD MODEL
# ============================================

print("Loading YOLO Model...")

model = YOLO(MODEL_PATH, task="detect")

print("Model Loaded Successfully!\n")

# ============================================
# SELECT IMAGE
# ============================================

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    initialdir=os.path.join(BASE_DIR, "Tungro"),
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
        ("All Files", "*.*")
    ]
)

if image_path == "":
    print("No image selected.")
    exit()

print("Selected Image:")
print(image_path)

# ============================================
# READ IMAGE
# ============================================

image = cv2.imread(image_path)

if image is None:
    print("Error: Cannot open image.")
    exit()

# ============================================
# RUN DETECTION
# ============================================

results = model(image, conf=0.5, verbose=False)

original = image.copy()
detected = results[0].plot()

# ============================================
# PRINT DETECTION
# ============================================

print("\n========== Detection Result ==========\n")

if len(results[0].boxes) == 0:
    print("No object detected.")

for box in results[0].boxes:

    cls = int(box.cls.item())
    conf = float(box.conf.item())

    print(f"Class      : {model.names[cls]}")
    print(f"Confidence : {conf:.2f}")
    print("-----------------------------------")

# ============================================
# RESIZE
# ============================================

DISPLAY_HEIGHT = 600

w1 = int(original.shape[1] * DISPLAY_HEIGHT / original.shape[0])
w2 = int(detected.shape[1] * DISPLAY_HEIGHT / detected.shape[0])

original = cv2.resize(original, (w1, DISPLAY_HEIGHT))
detected = cv2.resize(detected, (w2, DISPLAY_HEIGHT))

# ============================================
# ADD TITLES
# ============================================

cv2.putText(
    original,
    "Original Image",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.putText(
    detected,
    "Detection Result",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

# ============================================
# BLACK BORDER
# ============================================

BORDER = 8

original = cv2.copyMakeBorder(
    original,
    BORDER,
    BORDER,
    BORDER,
    BORDER,
    cv2.BORDER_CONSTANT,
    value=(0,0,0)
)

detected = cv2.copyMakeBorder(
    detected,
    BORDER,
    BORDER,
    BORDER,
    BORDER,
    cv2.BORDER_CONSTANT,
    value=(0,0,0)
)

# ============================================
# WHITE GAP
# ============================================

gap = np.full(
    (original.shape[0],25,3),
    255,
    dtype=np.uint8
)

combined = cv2.hconcat([original, gap, detected])

# ============================================
# SAVE RESULT
# ============================================

filename = os.path.basename(image_path)

save_path = os.path.join(RESULT_FOLDER, filename)

cv2.imwrite(save_path, combined)

print("\nSaved Result:")
print(save_path)

# ============================================
# DISPLAY
# ============================================

cv2.imshow("YOLO11 NCNN Detection", combined)

cv2.waitKey(0)

cv2.destroyAllWindows()