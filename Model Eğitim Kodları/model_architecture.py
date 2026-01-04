import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


def masked_gender_accuracy(y_true, y_pred):
    # y_true'yu float32'ye zorla (Hatanın çözümü burada)
    y_true = tf.cast(y_true, tf.float32)

    # Cinsiyet etiketi -1 olmayanları bul
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)

    # Tahminleri 0.5'e göre yuvarla ve doğruluğu hesapla
    correct_predictions = tf.cast(tf.equal(tf.round(y_pred), y_true), tf.float32)

    return tf.reduce_sum(correct_predictions * mask) / (tf.reduce_sum(mask) + 1e-7)


def masked_emotion_accuracy(y_true, y_pred):
    # Aynı garanti için burada da y_true'yu cast edelim
    y_true = tf.cast(y_true, tf.float32)

    # Duygu etiketi [0,0,0,0,0,0,0] olmayanları bul
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)

    # En yüksek olasılıklı sınıfı karşılaştır
    correct_predictions = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32)

    return tf.reduce_sum(correct_predictions * mask) / (tf.reduce_sum(mask) + 1e-7)

def masked_mse(y_true, y_pred):
    # -1 olanları 0'a çevir (sadece hesaplama hatası olmasın diye)
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    y_true_safe = tf.where(tf.equal(y_true, -1.0), tf.zeros_like(y_true), y_true)

    loss = tf.square(y_true_safe - y_pred)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-7)


def masked_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    # Satır bazlı kare farkı (reduction yapmadan)
    loss = tf.square(y_true - y_pred)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-7)


def masked_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_true (Batch,) -> (Batch, 1) boyutuna getir
    y_true = tf.reshape(y_true, [-1, 1])
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)

    # Sigmoid cross entropy formülü (y_true safe hale getirilerek)
    y_true_safe = tf.where(tf.equal(y_true, -1.0), tf.zeros_like(y_true), y_true)
    loss = tf.keras.losses.binary_crossentropy(y_true_safe, y_pred)
    loss = tf.reshape(loss, [-1, 1])

    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-7)


def masked_categorical_crossentropy(y_true, y_pred):
    # y_true'yu float32'ye cast et (dtype uyuşmazlığını önlemek için)
    y_true = tf.cast(y_true, tf.float32)

    # Duygu etiketleri one-hot yapıldığı için (tf.one_hot(-1, 7)),
    # etiketi olmayan (UTKFace) veriler [0,0,0,0,0,0,0] vektörüne dönüşür
    # Eğer vektörün toplamı 0'dan büyükse bu veri 'etiketli'dir (mask=1)
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)

    # Standart categorical crossentropy kaybını hesapla
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Sadece etiketi olan resimlerin hatasını topla ve etiketli veri sayısına böl
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-7)

class MultiTaskFaceModel:
    def __init__(self, input_shape=(224, 224, 3), num_emotions=7):
        self.input_shape = input_shape
        self.num_emotions = num_emotions
        self.model = None

    def build_base_model(self):
        """Temel ResNet50 modelini oluşturur"""
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )

        # Fine-tuning için katmanları ayarla
        base_model.trainable = True

        return base_model

    def create_model(self):
        """Multi-task modeli oluşturur"""
        print("Model oluşturuluyor...")

        # Giriş
        inputs = layers.Input(shape=self.input_shape, name='input_layer')

        # Temel model
        base_model = self.build_base_model()
        features = base_model(inputs)

        # Paylaşımlı katmanlar
        x = layers.Dense(512, activation='relu', name='shared_dense1')(features)
        x = layers.BatchNormalization(name='shared_bn1')(x)
        x = layers.Dropout(0.5, name='shared_dropout1')(x)

        x = layers.Dense(256, activation='relu', name='shared_dense2')(x)
        x = layers.BatchNormalization(name='shared_bn2')(x)
        x = layers.Dropout(0.3, name='shared_dropout2')(x)

        # 1. Çıktı: Yaş (Regresyon)
        age_branch = layers.Dense(128, activation='relu', name='age_dense1')(x)
        age_branch = layers.Dropout(0.2, name='age_dropout1')(age_branch)
        age_branch = layers.Dense(64, activation='relu', name='age_dense2')(age_branch)
        # model_architecture.py içinde yaş çıktı katmanını değiştir:
        age_output = layers.Dense(1, activation='sigmoid', name='age_output')(age_branch)

        # 2. Çıktı: Cinsiyet (Binary Classification)
        gender_branch = layers.Dense(128, activation='relu', name='gender_dense1')(x)
        gender_branch = layers.Dropout(0.2, name='gender_dropout1')(gender_branch)
        gender_branch = layers.Dense(64, activation='relu', name='gender_dense2')(gender_branch)
        gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(gender_branch)

        # 3. Çıktı: Duygu (Multi-class Classification)
        emotion_branch = layers.Dense(128, activation='relu', name='emotion_dense1')(x)
        emotion_branch = layers.Dropout(0.2, name='emotion_dropout1')(emotion_branch)
        emotion_branch = layers.Dense(64, activation='relu', name='emotion_dense2')(emotion_branch)
        emotion_output = layers.Dense(self.num_emotions,
                                      activation='softmax',
                                      name='emotion_output')(emotion_branch)

        # Modeli birleştir
        model = Model(
            inputs=inputs,
            outputs=[age_output, gender_output, emotion_output],
            name='multi_task_face_model'
        )

        self.model = model
        print("Model oluşturuldu!")
        return model

    def masked_gender_accuracy(y_true, y_pred):
        # Cinsiyet etiketi -1 olmayanları bul
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        # Tahminleri 0.5'e göre yuvarla ve doğruluğu hesapla
        correct_predictions = tf.cast(tf.equal(tf.round(y_pred), y_true), tf.float32)
        return tf.reduce_sum(correct_predictions * mask) / (tf.reduce_sum(mask) + 1e-7)

    def masked_emotion_accuracy(y_true, y_pred):
        # UTKFace için tf.one_hot(-1, 7) sonucu [0,0,0,0,0,0,0] döner.
        # Eğer toplam 0'dan büyükse bu geçerli bir etikettir (FER2013 verisidir).
        mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)

        # En yüksek olasılıklı sınıfı karşılaştır
        correct_predictions = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32)

        # Sadece etiketi olan resimlerin doğruluğunu döndür
        return tf.reduce_sum(correct_predictions * mask) / (tf.reduce_sum(mask) + 1e-7)

    def compile_model(self, learning_rate=0.00001):
        """Modeli derler"""
        if self.model is None:
            self.create_model()

        print("Model derleniyor...")

        # Kayıp fonksiyonları
        losses = {
            'age_output': masked_mse,
            'gender_output': masked_binary_crossentropy,
            'emotion_output': masked_categorical_crossentropy
        }

        # Loss ağırlıkları
        loss_weights = {
            'age_output': 5.0,
            'gender_output': 10.0,
            'emotion_output': 10.0
        }

        # Metrikler
        metrics = {
            'age_output': ['mae', 'mse'],
            'gender_output': [masked_gender_accuracy],
            'emotion_output': [masked_emotion_accuracy]
        }

        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Derle
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        print("Model derlendi!")
        return self.model

    def summary(self):
        """Model özetini gösterir"""
        if self.model:
            return self.model.summary()
        else:
            print("Model henüz oluşturulmadı!")
            return None

    def save_model(self, filepath='models/multi_task_model.h5'):
        """Modeli kaydeder"""
        if self.model:
            self.model.save(filepath)
            print(f"Model kaydedildi: {filepath}")
        else:
            print("Kaydedilecek model yok!")

    def load_model(self, filepath):
        """Modeli yükler"""
        self.model = keras.models.load_model(filepath)
        print(f"Model yüklendi: {filepath}")
        return self.model


# Test
if __name__ == "__main__":
    model_builder = MultiTaskFaceModel()
    model = model_builder.create_model()
    model_builder.compile_model()
    model_builder.summary()