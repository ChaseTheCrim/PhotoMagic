#!/usr/bin/env python3
"""
Model eğitim script'i
Kullanım: python train_model.py --epochs 50 --batch_size 16
"""

import os
import sys
import argparse
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Bellek Yönetimi: Aktif (Dinamik kullanım)")
    except RuntimeError as e:
        print(f"GPU Ayar Hatası: {e}")

# Proje kök dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessor import DataPreprocessor
from src.model_architecture import MultiTaskFaceModel
from src.model_trainer import ModelTrainer


def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser(description='Multi-task Face Model Eğitimi')
    parser.add_argument('--epochs', type=int, default=15, help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch boyutu')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Öğrenme oranı')
    parser.add_argument('--data_dir', type=str, default='data', help='Veri dizini')

    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-TASK YÜZ ANALİZİ MODELİ EĞİTİMİ")
    print("=" * 60)

    # 1. Veri hazırlama
    print("\n1. VERİ HAZIRLANIYOR...")
    preprocessor = DataPreprocessor(target_size=(224, 224))

    # Veri yolları
    utkface_path = os.path.join(args.data_dir, 'UTKFace')
    fer2013_path = os.path.join(args.data_dir, 'fer2013')

    if not os.path.exists(utkface_path):
        print(f"Uyarı: UTKFace dizini bulunamadı: {utkface_path}")
        print("Lütfen UTKFace datasetini indirip buraya koyun.")
        return

    if not os.path.exists(fer2013_path):
        print(f"Uyarı: FER-2013 CSV dosyası bulunamadı: {fer2013_path}")
        print("Lütfen fer2013.csv'yi indirip buraya koyun.")
        return

    # Datasetleri hazırla
    train_dataset, val_dataset, test_dataset = preprocessor.prepare_datasets(
        utkface_path=utkface_path,
        fer2013_path=fer2013_path,
        test_size=0.2,
        val_size=0.1
    )

    # 2. Model oluşturma
    print("\n2. MODEL OLUŞTURULUYOR...")
    model_builder = MultiTaskFaceModel(input_shape=(224, 224, 3), num_emotions=7)
    model = model_builder.compile_model(learning_rate=args.learning_rate)
    model_builder.summary()

    # 3. Model eğitimi
    print("\n3. MODEL EĞİTİLİYOR...")
    trainer = ModelTrainer(model, model_name="multi_task_face")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs
    )
    print("\nEĞİTİM TAMAMLANDI!")

if __name__ == "__main__":
    main()