import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


class FacePredictor:
    def __init__(self, model_path=None, model=None):
        if model:
            self.model = model
        elif model_path:
            self.model = keras.models.load_model(model_path)
        else:
            raise ValueError("Model veya model yolu gereklidir!")

        self.emotion_labels = {
            0: 'Kızgın (Angry)',
            1: 'İğrenme (Disgust)',
            2: 'Korku (Fear)',
            3: 'Mutlu (Happy)',
            4: 'Üzgün (Sad)',
            5: 'Şaşkın (Surprise)',
            6: 'Nötr (Neutral)'
        }

        print("FacePredictor başarıyla yüklendi!")

    def preprocess_image(self, image, target_size=(224, 224)):
        """Görüntüyü model için hazırlar"""
        # Görüntüyü kopyala
        img = image.copy()

        # Boyutlandır
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)

        # BGR'dan RGB'ye çevir
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize et
        img = img.astype('float32') / 255.0

        # Batch boyutu ekle
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image):
        """Tek bir görüntü üzerinde tahmin yapar"""
        # Ön işle
        processed_img = self.preprocess_image(image)

        # Tahmin
        age_pred, gender_pred, emotion_pred = self.model.predict(processed_img, verbose=0)

        # Sonuçları işle
        age = int(age_pred[0][0])
        age = max(0, min(100, age))  # 0-100 arasına sınırla

        gender_score = gender_pred[0][0]
        gender = "Kadın" if gender_score > 0.5 else "Erkek"
        gender_confidence = gender_score if gender_score > 0.5 else 1 - gender_score

        emotion_idx = np.argmax(emotion_pred[0])
        emotion = self.emotion_labels.get(emotion_idx, "Bilinmeyen")
        emotion_confidence = emotion_pred[0][emotion_idx]

        # Sonuçları paketle
        results = {
            'age': age,
            'gender': gender,
            'gender_confidence': float(gender_confidence),
            'emotion': emotion,
            'emotion_confidence': float(emotion_confidence),
            'raw_predictions': {
                'age': float(age_pred[0][0]),
                'gender': float(gender_score),
                'emotions': emotion_pred[0].tolist()
            }
        }

        return results

    def predict_batch(self, images):
        """Toplu tahmin yapar"""
        results = []

        for img in images:
            try:
                result = self.predict(img)
                results.append(result)
            except Exception as e:
                print(f"Tahmin hatası: {e}")
                results.append(None)

        return results

    def print_results(self, results):
        """Tahmin sonuçlarını güzel bir şekilde yazdırır"""
        print("\n" + "=" * 50)
        print("TAHMİN SONUÇLARI")
        print("=" * 50)

        print(f"Yaş: {results['age']} yıl")
        print(f"Cinsiyet: {results['gender']} (%{results['gender_confidence'] * 100:.1f} güven)")
        print(f"Duygu: {results['emotion']} (%{results['emotion_confidence'] * 100:.1f} güven)")
        print("=" * 50)

    def predict_from_file(self, image_path):
        """Dosyadan görüntü yükleyip tahmin yapar"""
        # Görüntüyü yükle
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")

        print(f"Görüntü yüklendi: {image_path}")
        print(f"Boyut: {image.shape}")

        # Tahmin yap
        results = self.predict(image)

        # Görselleştir
        self.visualize_prediction(image, results)

        return results

    def visualize_prediction(self, image, results):
        """Tahmin sonuçlarını görselleştirir"""
        # Görüntüyü kopyala
        img_display = image.copy()

        # BGR'dan RGB'ye çevir
        if len(img_display.shape) == 3:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        # Bilgileri yaz
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # Beyaz
        thickness = 2

        # Metinleri hazırla
        texts = [
            f"Yaş: {results['age']}",
            f"Cinsiyet: {results['gender']} (%{results['gender_confidence'] * 100:.1f})",
            f"Duygu: {results['emotion'].split()[0]} (%{results['emotion_confidence'] * 100:.1f})"
        ]

        # Metinleri ekle
        y_offset = 30
        for i, text in enumerate(texts):
            cv2.putText(img_display, text, (10, y_offset * (i + 1)),
                        font, font_scale, color, thickness)

        # Görüntüyü göster
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.imshow(img_display)
        plt.axis('off')
        plt.title("Yüz Analizi Sonuçları")
        plt.show()


# Test
if __name__ == "__main__":
    # Örnek kullanım
    predictor = FacePredictor(model_path="models/final_model.h5")

    # Test için örnek görüntü
    test_image = np.random.rand(224, 224, 3).astype('float32')

    results = predictor.predict(test_image)
    predictor.print_results(results)