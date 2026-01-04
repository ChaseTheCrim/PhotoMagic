import os
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


class ModelTrainer:
    def __init__(self, model, model_name="face_model"):
        self.model = model
        self.model_name = model_name
        self.history = None

    def create_callbacks(self, checkpoint_dir='models/checkpoints'):
        """Eğitim callback'lerini oluşturur"""

        # Klasörü oluştur
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Checkpoint dosya yolu
        checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_best.h5')

        callbacks = [
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1,
                save_weights_only=False
            ),

            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),

            # Learning rate scheduler
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard
            # keras.callbacks.TensorBoard(
            #     log_dir='logs',
            #     histogram_freq=1,
            #     update_freq='batch'
            # )
        ]

        print(f"Checkpoint dosyası: {checkpoint_path}")
        return callbacks

    def train(self, train_dataset, val_dataset, epochs=50):
        """Modeli eğitir"""

        print("=" * 50)
        print("MODEL EĞİTİMİ BAŞLIYOR")
        print("=" * 50)

        # Callback'leri oluştur
        callbacks = self.create_callbacks()

        # Modeli eğit
        print(f"Eğitim başlıyor... Toplam epoch: {epochs}")

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("=" * 50)
        print("MODEL EĞİTİMİ TAMAMLANDI")
        print("=" * 50)

        return self.history

    def plot_history(self, save_path=None):
        """Eğitim geçmişini görselleştirir"""
        if self.history is None:
            print("Henüz eğitim yapılmadı!")
            return

        history = self.history.history

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        # Yaş Loss
        axes[0, 0].plot(history['age_output_loss'], label='Train')
        axes[0, 0].plot(history['val_age_output_loss'], label='Validation')
        axes[0, 0].set_title('Age Loss (MSE)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Cinsiyet Loss
        axes[0, 1].plot(history['gender_output_loss'], label='Train')
        axes[0, 1].plot(history['val_gender_output_loss'], label='Validation')
        axes[0, 1].set_title('Gender Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Duygu Loss
        axes[0, 2].plot(history['emotion_output_loss'], label='Train')
        axes[0, 2].plot(history['val_emotion_output_loss'], label='Validation')
        axes[0, 2].set_title('Emotion Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Yaş MAE
        axes[1, 0].plot(history['age_output_mae'], label='Train')
        axes[1, 0].plot(history['val_age_output_mae'], label='Validation')
        axes[1, 0].set_title('Age MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Cinsiyet Accuracy
        axes[1, 1].plot(history['gender_output_accuracy'], label='Train')
        axes[1, 1].plot(history['val_gender_output_accuracy'], label='Validation')
        axes[1, 1].set_title('Gender Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Duygu Accuracy
        axes[1, 2].plot(history['emotion_output_accuracy'], label='Train')
        axes[1, 2].plot(history['val_emotion_output_accuracy'], label='Validation')
        axes[1, 2].set_title('Emotion Accuracy')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Learning Rate
        if 'lr' in history:
            axes[2, 0].plot(history['lr'])
            axes[2, 0].set_title('Learning Rate')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('LR')
            axes[2, 0].grid(True, alpha=0.3)

        # Total Loss
        axes[2, 1].plot(history['loss'], label='Train')
        axes[2, 1].plot(history['val_loss'], label='Validation')
        axes[2, 1].set_title('Total Loss')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # Boş subplot'u gizle
        axes[2, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")

        plt.show()

    def save_final_model(self, path='models/final_model.h5'):
        """Final modeli kaydeder"""
        self.model.save(path)
        print(f"Final model kaydedildi: {path}")

    def evaluate(self, test_dataset):
        """Test setinde değerlendirir"""
        if self.model is None:
            print("Model yüklenmedi!")
            return None

        print("=" * 50)
        print("MODEL DEĞERLENDİRME")
        print("=" * 50)

        results = self.model.evaluate(test_dataset, verbose=1)

        # Sonuçları göster
        print("\nDeğerlendirme Sonuçları:")
        print("-" * 30)

        metrics_names = self.model.metrics_names
        for name, value in zip(metrics_names, results):
            print(f"{name}: {value:.4f}")

        return results


# Test
if __name__ == "__main__":
    # Örnek kullanım
    from model_architecture import MultiTaskFaceModel

    model_builder = MultiTaskFaceModel()
    model = model_builder.compile_model()

    trainer = ModelTrainer(model)
    print("ModelTrainer başarıyla oluşturuldu!")