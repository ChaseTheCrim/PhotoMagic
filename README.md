# 📸 PhotoMagic AI (v1.0)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Framework](https://img.shields.io/badge/GUI-PySide6-41CD52?style=for-the-badge&logo=qt&logoColor=white)
![AI Engine](https://img.shields.io/badge/AI-TensorFlow%20%7C%20OpenCV-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/github/license/ChaseTheCrim/PhotoMagic?style=for-the-badge)

**PhotoMagic** is a production-ready desktop biometric analysis tool developed by the **PhotoSynth Team**. It features a **Hybrid AI Architecture** capable of real-time Age, Gender, Emotion recognition, and Face Skeleton tracking, wrapped in a modern PySide6 interface.

---

## 🚀 Key Features

### 🧠 Hybrid AI Engine
* **Custom Multi-Task Models:** Runs specialized ResNet50-based models trained on UTKFace & FER2013 datasets for high-accuracy predictions.
* **Modular Backend:** Architecture allows switching between our custom models and `DeepFace` library for benchmarking purposes.
* **Smart Caching:** Optimized inference logic to maintain high FPS during simultaneous video processing.

### 🛡️ Fail-Safe Data Integrity
* **Smart Save:** Integrated image quality assessment algorithms (blur & noise detection) to prevent low-quality data from corrupting the `FaceDatabase`.
* **Logic Redundancy:** Prevents duplicate entries and handles camera disconnections gracefully.

### ⚡ Real-Time Capabilities
* **Biometric Analysis:** Simultaneous prediction of Age, Gender, and Emotion.
* **Skeleton Tracking:** 468-point face mesh tracking powered by MediaPipe.
* **Precision Preprocessing:** Implements a 15% margin cropping algorithm to match training conditions for better real-world accuracy.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **GUI Framework:** PySide6 (Qt)
* **Deep Learning:** TensorFlow / Keras (Custom Models)
* **Computer Vision:** OpenCV, MediaPipe, DeepFace
* **Data Handling:** NumPy, Pandas

---

## 📂 Project Structure

```text
PhotoMagic/
├── AI_Core.py            # Hybrid AI Engine (Logic & Inference)
├── main.py               # Main Application Entry & GUI Event Loop
├── mainlib.py            # Helper Functions (Image Processing, Fail-Safe)
├── model_architecture.py # Custom ResNet50 Model Architecture
├── qtGUI.ui              # Qt Designer Interface File
├── models/               # Pre-trained .h5 Models (See Releases)
├── FaceDatabase/         # Encrypted Face Embeddings Storage
└── requirements.txt      # Project Dependencies
