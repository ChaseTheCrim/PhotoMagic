import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def load_fer2013_paths(self, fer2013_dir):
        """Resimleri OKUMAZ, sadece dosya yollarını ve etiketleri toplar."""
        emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
        paths, emotions = [], []

        for split in ['train', 'test']:
            split_dir = os.path.join(fer2013_dir, split)
            if not os.path.exists(split_dir): continue
            for emotion_name, emotion_id in emotion_labels.items():
                emotion_dir = os.path.join(split_dir, emotion_name)
                if not os.path.exists(emotion_dir): continue
                for img_file in os.listdir(emotion_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        paths.append(os.path.join(emotion_dir, img_file))
                        emotions.append(emotion_id)
        return np.array(paths), np.array(emotions)

    def load_utkface_paths(self, dataset_path):
        """UTKFace için dosya yollarını ve yaş/cinsiyet verilerini toplar."""
        paths, ages, genders = [], [], []
        for filename in os.listdir(dataset_path):
            if filename.endswith('.jpg'):
                parts = filename.split('_')
                if len(parts) >= 2:
                    paths.append(os.path.join(dataset_path, filename))
                    ages.append(int(parts[0]))
                    genders.append(int(parts[1]))
        return np.array(paths), np.array(ages), np.array(genders)

    def load_combined_dataset(self, utkface_path, fer2013_path):
        """Sadece dosya yollarını birleştirir. RAM kullanımı minimaldir."""
        print("1. Veri yolları toplanıyor...")
        utk_p, utk_a, utk_g = self.load_utkface_paths(utkface_path)
        fer_p, fer_e = self.load_fer2013_paths(fer2013_path)

        utk_a = utk_a.astype(np.float32) / 100.0 # UTKFace'den gelen gerçek yaşları normalize ediyoruz. (0.0 - 1.0 arası)

        # Eksik etiketleri mantıklı varsayılanlarla doldur (Broadcast)
        utk_e = np.full((len(utk_p),), -1)  # UTK -> Nötr Duygu
        fer_a = np.full((len(fer_p),), -1.0)  # FER -> 25 Yaş
        fer_g = np.full((len(fer_p),), -1.0)  # FER -> Belirsiz Cinsiyet

        # Tüm yolları ve etiketleri birleştir
        all_paths = np.concatenate([utk_p, fer_p])
        all_ages = np.concatenate([utk_a, fer_a])
        all_genders = np.concatenate([utk_g, fer_g])
        all_emotions = np.concatenate([utk_e, fer_e])

        return all_paths, all_ages, all_genders, all_emotions

    def _parse_function(self, path, age, gender, emotion):
        """Eğitim sırasında resmi diskten okuyan yardımcı fonksiyon."""
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.target_size)
        image = tf.image.random_flip_left_right(image)

        # Modelin beklediği output formatı: {'output_name': value}
        return image, {
            'emotion_output': emotion,
            'age_output': age,
            'gender_output': gender
        }

    def create_tf_dataset(self, paths, ages, genders, emotions, batch_size=32, shuffle=True):
        """TensorFlow veri akış hattı oluşturur."""
        # Duyguları one-hot yap (7 sınıf)
        emotions_oh = tf.one_hot(emotions, 7)

        # 2. Cinsiyetleri (N, 1) boyutuna getir
        genders_reshaped = np.array(genders).reshape(-1, 1)

        dataset = tf.data.Dataset.from_tensor_slices((paths, ages, genders_reshaped, emotions_oh))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        # Ryzen 9'un çekirdeklerini kullanarak paralel okuma yap
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def prepare_datasets(self, utkface_path, fer2013_path, test_size=0.2, val_size=0.1):
        """Tüm veri yollarını toplar, böler ve TF Dataset'lerini oluşturur."""
        print("=" * 50)
        print("VERİ HAZIRLAMA (STREAMING MODU)")
        print("=" * 50)

        # 1. Sadece dosya yollarını ve etiketleri al (RAM dostu)
        all_paths, all_ages, all_genders, all_emotions = self.load_combined_dataset(
            utkface_path, fer2013_path
        )

        # 2. Veri bölme (Train / Validation / Test)
        print("\nVeri bölme işlemi yapılıyor...")

        # Önce Test setini ayır
        paths_temp, paths_test, age_temp, age_test, gen_temp, gen_test, emo_temp, emo_test = train_test_split(
            all_paths, all_ages, all_genders, all_emotions,
            test_size=test_size, random_state=42, stratify=all_emotions
        )

        # Kalanı Train ve Validation olarak ayır
        val_ratio = val_size / (1 - test_size)
        paths_train, paths_val, age_train, age_val, gen_train, gen_val, emo_train, emo_val = train_test_split(
            paths_temp, age_temp, gen_temp, emo_temp,
            test_size=val_ratio, random_state=42, stratify=emo_temp
        )

        print(f"Eğitim: {len(paths_train)} | Doğrulama: {len(paths_val)} | Test: {len(paths_test)}")

        # 3. TensorFlow Dataset'lerini oluştur
        # RTX 5070 Ti için batch_size 64 idealdir
        train_dataset = self.create_tf_dataset(paths_train, age_train, gen_train, emo_train, batch_size=64,
                                               shuffle=True)
        val_dataset = self.create_tf_dataset(paths_val, age_val, gen_val, emo_val, batch_size=64, shuffle=False)
        test_dataset = self.create_tf_dataset(paths_test, age_test, gen_test, emo_test, batch_size=64, shuffle=False)

        print("=" * 50)
        print("VERİ HAZIRLAMA TAMAMLANDI - EĞİTİME HAZIR")
        print("=" * 50)

        return train_dataset, val_dataset, test_dataset