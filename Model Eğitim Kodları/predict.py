import tensorflow as tf
import cv2
import numpy as np
import os


def predict_my_photo(image_path, model_path):
    # 1. Modeli Yükle
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        return

    print("Model yükleniyor...")
    model = tf.keras.models.load_model(model_path, compile=False)

    # 2. Resmi Yükle ve Ön İşle
    if not os.path.exists(image_path):
        print(f"Hata: Resim bulunamadı: {image_path}")
        return

    img = cv2.imread(image_path)
    # OpenCV'nin hazır yüz bulma algoritmasını yüklüyoruz
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti yapıyoruz (scaleFactor ve minNeighbors değerleri hassasiyet içindir)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # En büyük yüzü al (genelde ilkidir)
        (x, y, w, h) = faces[0]

        # UTKFace stiline uyması için yüzü biraz pay bırakarak (%15 pay) kırpıyoruz
        margin = int(w * 0.15)
        y_start = max(0, y - margin)
        y_end = min(img.shape[0], y + h + margin)
        x_start = max(0, x - margin)
        x_end = min(img.shape[1], x + w + margin)

        face_img = img[y_start:y_end, x_start:x_end]
        print(f"Yüz başarıyla tespit edildi ve kırpıldı (Konum: {x},{y})")
    else:
        # Yüz bulunamazsa resmin tamamını kullan
        face_img = img
        print("Uyarı: Yüz tespit edilemedi, resmin tamamı kullanılıyor.")

    # Ön İşlem
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) # Modelin beklediği boyut
    img_array = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Tahmin Yap
    print("Tahmin yapılıyor...")
    predictions = model.predict(img_batch)

    # Çıktıları ayıkla (Modeldeki isimlere göre: age, gender, emotion)
    # Model çıktı sırası: [age_output, gender_output, emotion_output]
    age_pred = predictions[0][0][0]
    gender_pred = predictions[1][0][0]
    emotion_probs = predictions[2][0]

    print(f"DEBUG - Ham Yas Tahmini (0-1 arasi): {age_pred}")
    print(f"DEBUG - Ham Cinsiyet Tahmini: {gender_pred}")

    # Duygu etiketleri
    emotions = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']
    detected_emotion = emotions[np.argmax(emotion_probs)]

    # Cinsiyet yorumlama
    gender = "Erkek" if gender_pred > 0.5 else "Kadin"
    gender_conf = gender_pred if gender_pred > 0.5 else 1 - gender_pred

    # 4. Sonuçları Yazdır
    print("\n" + "=" * 30)
    print("      MODEL TAHMİN SONUÇLARI")
    print("=" * 30)
    print(f"Duygu    : {detected_emotion} ({np.max(emotion_probs) * 100:.2f}%)")
    print(f"Yaş      : {int(age_pred * 100)} yaş")
    print(f"Cinsiyet : {gender} ({gender_conf * 100:.2f}%)")
    print("=" * 30)


if __name__ == "__main__":
    # BURAYI DÜZENLEMEYİ UNUTMA: Test etmek istediğin resmin adını yaz
    my_photo = "deneme7.jpeg"
    model_file = "models/multi_task_face_best.h5"

    predict_my_photo(my_photo, model_file)