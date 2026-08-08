# 🌾 Rice Plant Disease Detection Using YOLO and NCNN

## 📌 Project Overview

This project is an **AI-based rice plant disease detection system** designed to identify diseases from images of rice plants.

The project uses a trained **YOLO object detection model**, exported to **NCNN format**, to perform efficient disease detection. The NCNN model can be used for lightweight and fast inference on computers and embedded systems.

The current dataset and testing setup primarily focuses on **Tungro disease in rice plants**.

---

## 🎯 Objectives

The main objectives of this project are:

* Detect diseases in rice plant images using deep learning.
* Identify **Tungro disease** from plant images.
* Train and test an object detection model.
* Export the trained model into **NCNN format**.
* Perform inference using Python from the Windows Command Prompt.
* Provide a foundation for deploying the disease detection system on resource-constrained hardware.

---

## 🧠 Technologies Used

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Model testing and image processing   |
| YOLO         | Object detection model               |
| NCNN         | Lightweight neural-network inference |
| OpenCV       | Image processing                     |
| Ultralytics  | YOLO model training/export           |
| Google Colab | Model training                       |
| Git/GitHub   | Project version control              |

---

## 📂 Project Structure

The repository contains the following major components:

```text
plant traning/
│
├── Tungro/
│   └── Rice plant images
│
├── labels/
│   └── YOLO annotation files
│
├── best_ncnn_model/
│   └── Trained NCNN model files
│
├── test.py
├── test2.py
├── test3.py
│
└── README.md
```

### 📁 `Tungro`

This folder contains the rice plant images used for testing and/or dataset preparation.

The images primarily represent rice plants affected by **Tungro disease**.

### 📁 `labels`

This folder contains the corresponding **YOLO-format annotation files**.

Each `.txt` label file contains the object class and bounding-box information required for object detection.

Typical YOLO annotation format:

```text
class_id x_center y_center width height
```

The coordinates are normalized between `0` and `1`.

### 📁 `best_ncnn_model`

This folder contains the trained model exported in **NCNN format**.

The NCNN model is used during inference to detect disease-related objects in input images.

### 🐍 `test.py`

Python script used for testing the trained model.

### 🐍 `test2.py`

Python script used for image-based model testing and processing.

### 🐍 `test3.py`

Additional testing/inference implementation for the trained model.

---

# 🔬 Disease Detection

## Tungro Disease

**Rice Tungro disease** is a major disease affecting rice plants.

The detection system analyzes an input image and uses the trained neural-network model to identify the target disease/object.

The general processing pipeline is:

```text
Rice Plant Image
       ↓
Image Preprocessing
       ↓
NCNN YOLO Model
       ↓
Object Detection
       ↓
Bounding Box + Confidence
       ↓
Disease Detection Result
```

---

# ⚙️ Model

The trained YOLO model has been exported to **NCNN format** for lightweight inference.

The model directory is:

```text
best_ncnn_model/
```

NCNN is particularly useful when deploying neural-network models on systems with limited computational resources.

The model can be used with Python through the NCNN inference framework.

---

# 💻 Running the Model on Windows

## 1. Install Python

Make sure Python is installed on your computer.

Check the installation:

```cmd
python --version
```

or:

```cmd
python3 --version
```

---

## 2. Install Required Libraries

Install the required Python packages:

```cmd
pip install opencv-python
pip install ncnn
pip install numpy
```

If the project uses additional packages in the testing scripts, install them as required.

---

## 3. Open the Project Folder

Open Command Prompt and navigate to the project directory:

```cmd
cd /d "D:\College\internship\COE\plant traning"
```

---

## 4. Run the Testing Code

For example:

```cmd
python test.py
```

or:

```cmd
python test2.py
```

or:

```cmd
python test3.py
```

Use the script appropriate for the required testing method.

---

# 🖼️ Image Processing Workflow

The testing process follows this general workflow:

```text
Select/Input Rice Plant Image
             ↓
       Read Image
             ↓
     Image Preprocessing
             ↓
       NCNN Inference
             ↓
      YOLO Detection
             ↓
   Detect Disease/Object
             ↓
Display Detection Result
```

The output generally contains the detected region along with the corresponding class and confidence score.

---

# 📊 Dataset

The dataset contains images of rice plants used for training/testing the disease detection model.

The dataset includes:

* Rice plant images
* Disease images
* YOLO annotation files
* Class information
* Bounding-box annotations

The current project primarily focuses on **Tungro disease detection**.

---

# 🏷️ YOLO Label Format

The annotation files use the YOLO format.

Example:

```text
0 0.512 0.483 0.321 0.456
```

Where:

```text
0       → Class ID
0.512   → X-center
0.483   → Y-center
0.321   → Bounding-box width
0.456   → Bounding-box height
```

All coordinates are normalized relative to the image dimensions.

---

# 🚀 Future Development

The project can be further developed in several directions:

* Add more rice plant diseases.
* Increase the size and diversity of the dataset.
* Improve detection accuracy.
* Optimize the NCNN model for embedded devices.
* Deploy the model on Raspberry Pi.
* Integrate a camera for real-time detection.
* Develop real-time rice disease monitoring.
* Add automatic disease classification and reporting.
* Develop a mobile or web-based interface.

---

# 🔧 Possible Deployment

The lightweight NCNN model makes the project suitable for future deployment on embedded platforms such as:

```text
Camera
   ↓
Raspberry Pi / Embedded Computer
   ↓
NCNN YOLO Model
   ↓
Rice Plant Detection
   ↓
Disease Identification
```

This can eventually be developed into a **real-time autonomous rice plant disease monitoring system**.

---

# 📌 Project Status

**Current Status:** Model training and testing completed/in progress.

The current implementation includes:

* ✅ Rice plant image dataset
* ✅ YOLO annotations
* ✅ Trained detection model
* ✅ NCNN model export
* ✅ Python inference scripts
* ✅ Windows Command Prompt testing
* 🔄 Further optimization and deployment

---

# 👨‍💻 Project Purpose

This project was developed as part of an **AI-based plant disease detection and image-processing study**, with the goal of applying computer vision and deep learning techniques to agricultural applications.

The project demonstrates the complete workflow from:

**Dataset → Annotation → Model Training → NCNN Export → Python Inference → Disease Detection**

---

## 📜 License

This project is intended for **educational, research, and development purposes**.

Please verify dataset and model licensing before using the project for commercial applications.

---

## ⭐ Acknowledgement

This project uses open-source technologies including **YOLO, NCNN, Python, OpenCV, and Ultralytics**.

The project focuses on applying these technologies to the detection of diseases in rice plants.
