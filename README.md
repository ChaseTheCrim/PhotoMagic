# PhotoMagic (v1.0.0)

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
```
## ⚙️ Installation (Kurulum)

### Option A: For Users (Son Kullanıcılar İçin)
Python veya kodlarla uğraşmanıza gerek yok. Hazır paketlenmiş sürümü kullanabilirsiniz:

1.  Bu sayfanın sağ tarafındaki **[Releases](../../releases)** kısmına gidin.
2.  En son yayınlanan `PhotoMagic_v1.0_Windows.zip` dosyasını indirin.
3.  Dosyayı zipten çıkarın.
4.  Klasör içindeki **`PhotoMagic.exe`** dosyasına çift tıklayın ve çalıştırın.

### Option B: For Developers (Geliştiriciler İçin)
Kaynak kodlarını incelemek veya katkıda bulunmak isterseniz:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [https://github.com/ChaseTheCrim/PhotoMagic.git](https://github.com/ChaseTheCrim/PhotoMagic.git)
    cd PhotoMagic
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv venv
    # Windows için:
    venv\Scripts\activate
    ```

3.  **Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Uygulamayı Başlatın:**
    ```bash
    python main.py
    ```
    *(Not: Kaynak koddan çalıştırırken `models/` klasörünün içinde `.h5` model dosyalarının olduğundan emin olun.)*

---

## 🎮 Usage Guide (Kullanım Kılavuzu)

Uygulama arayüzü 3 ana panelden oluşur:

### 1. Ayarlar Paneli (Settings Panel)
Görüntü kalitesini anlık olarak ayarlayabilirsiniz:
* **Parlaklık & Kontrast:** Görüntü ışığını ve renk dengesini optimize eder.
* **Bulanıklık (Blur):** Görüntüyü yumuşatır (Noise azaltmak için kullanışlıdır).
* **Keskinlik (Sharpening):** Detayları belirginleştirir.

### 2. AI Modları & Efektler
* **✅ Yüz İskeleti:** 468 noktalı yüz ağını (mesh) gerçek zamanlı çizer.
* **✅ Yüz Tanıma:** Veritabanındaki kayıtlı kişileri (Ahmet, Ayşe vb.) tanır.
* **✅ Yaş ve Cinsiyet:** Tahmini yaş ve cinsiyet bilgisini yüzün yanına yazar.
* **🎨 Efektler:** Gri Tonlama, Negatif ve Kenar Algılama gibi filtreleri uygular.

### 3. Kontrol Butonları
* **Webcam:** Kamerayı başlatır.
* **Yükle:** Bilgisayardan statik bir fotoğraf yükleyerek analiz yapmanızı sağlar.
* **Kaydet:**
    * *Normal Mod:* Ekran görüntüsünü `Results/` klasörüne kaydeder.
    * *İskelet Modu:* Kişiyi **Yüz Veritabanına (FaceDatabase)** kaydetmek için kayıt penceresini açar.
* **Sıfırla:** Tüm ayarları ve efektleri varsayılan hale getirir.

---

## 📄 License & Copyright

**PhotoMagic** is developed by the **PhotoSynth Team**.
Distributed under the **MIT License**.

Bu proje açık kaynaklıdır ve eğitim/portföy amaçlı geliştirilmiştir. Ticari kullanım için lisans dosyasını inceleyiniz.

Copyright © 2026 PhotoSynth Team. All Rights Reserved.
