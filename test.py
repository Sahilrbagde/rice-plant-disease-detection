from ultralytics import YOLO
import cv2
import numpy as np

# ============================================
# LOAD MODEL
# ============================================
model = YOLO(
    r"D:\College\internship\plant traning\best_ncnn_model",
    task="detect"
)

# ============================================
# IMAGE PATH
# ============================================
image_path = r"D:\College\internship\plant traning\leaf.jpg"   # <-- Change this

# ============================================
# READ IMAGE
# ============================================
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# ============================================
# RUN DETECTION
# ============================================
results = model(image, conf=0.5, verbose=False)

# Original copy
original = image.copy()

# Detection image
detected = results[0].plot()

# ============================================
# PRINT DETECTIONS
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
# RESIZE IMAGES
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
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.putText(
    detected,
    "Detection Result",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# ============================================
# ADD BLACK BORDER
# ============================================
BORDER = 8

original = cv2.copyMakeBorder(
    original,
    BORDER,
    BORDER,
    BORDER,
    BORDER,
    cv2.BORDER_CONSTANT,
    value=(0, 0, 0)
)

detected = cv2.copyMakeBorder(
    detected,
    BORDER,
    BORDER,
    BORDER,
    BORDER,
    cv2.BORDER_CONSTANT,
    value=(0, 0, 0)
)

# ============================================
# WHITE GAP BETWEEN IMAGES
# ============================================
gap = np.full(
    (original.shape[0], 25, 3),
    255,
    dtype=np.uint8
)

# ============================================
# COMBINE IMAGES
# ============================================
combined = cv2.hconcat([original, gap, detected])

# ============================================
# SAVE RESULT
# ============================================
save_path = r"D:\College\internship\plant traning\comparison_result.jpg"

cv2.imwrite(save_path, combined)

print("\nSaved Result To:")
print(save_path)

# ============================================
# DISPLAY
# ============================================
cv2.imshow("YOLO11 NCNN Comparison", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()