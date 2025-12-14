import cv2
import numpy as np
import mediapipe as mp
import os
import time
import face_recognition  # Yüz tanıma kütüphanesini ekledik


class MiniPhotoshop:
    def __init__(self):
        # MediaPipe Selfie Segmentation modelini yükle
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1)  # model_selection: 0 genel, 1 daha hızlı

        # İşlenmiş görüntüyü saklamak için
        self.processed_image = None
        self.captured_image = None

        # --- YENİ EKLENEN YÜZ TANIMA DEĞİŞKENLERİ ---
        self.face_rec_enabled = 0  # 0: Kapalı, 1: Açık
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()  # Yüzleri yükle
        # ---------------------------------------------

        # Trackbar değerleri için değişkenler
        self.brightness = 50
        self.contrast = 50
        self.blur_intensity = 0
        self.sharpness = 0
        self.grayscale = 0
        self.inversion = 0
        self.portrait_mode = 0
        self.portrait_blur = 5

        # Orijinal görüntüyü saklamak için
        self.original_image = None

        # Video yakalama için
        self.cap = None
        self.webcam_mode = False

        # Çıkış klasörü
        self.output_dir = "islenmis_foto"
        self.create_output_dir()

        # Pencere adı
        self.window_name = "Mini Photoshop & Face Rec"

        # FPS takibi için
        self.prev_time = 0
        self.fps = 0

    def load_known_faces(self):
        # --- YENİ FONKSİYON: Tanınacak yüzleri yükler ---
        target_file = "ornek_kisi.jpg"
        if os.path.exists(target_file):
            print(f"Yüz tanıma verisi yükleniyor: {target_file}...")
            try:
                image = face_recognition.load_image_file(target_file)
                encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append("Hedef Kisi")  # İsim buradan değiştirilebilir
                print("Yüz verisi başarıyla işlendi.")
            except IndexError:
                print("HATA: Fotoğrafta yüz bulunamadı!")
            except Exception as e:
                print(f"HATA: {e}")
        else:
            print(f"UYARI: '{target_file}' bulunamadı. Yüz tanıma çalışmayacak.")
            print("Lütfen proje klasörüne 'ornek_kisi.jpg' adında bir fotoğraf ekleyin.")

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Görüntü yüklenemedi: {image_path}")
            return False
        self.processed_image = self.original_image.copy()
        self.webcam_mode = False
        return True

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Webcam açılamadı!")
            return False
        self.webcam_mode = True
        ret, frame = self.cap.read()
        if ret:
            self.original_image = frame
            self.processed_image = frame.copy()
            self.captured_image = None
            return True
        return False

    def get_next_webcam_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False
        ret, frame = self.cap.read()
        if ret:
            self.original_image = frame
            return True
        return False

    def capture_current_frame(self):
        if self.processed_image is not None:
            self.captured_image = self.processed_image.copy()
            return True
        return False

    def save_image(self, image=None):
        if image is None:
            image = self.processed_image
        if image is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"webcam_{timestamp}.jpg" if self.webcam_mode else f"foto_{timestamp}.jpg"
            file_path = os.path.join(self.output_dir, filename)
            success = cv2.imwrite(file_path, image)
            if success:
                print(f"Görüntü kaydedildi: {file_path}")
                return True
            else:
                print(f"Görüntü kaydedilemedi: {file_path}")
                return False
        return False

    # --- FİLTRE FONKSİYONLARI ---
    def apply_brightness_contrast(self, image, brightness=50, contrast=50):
        # Ayarları ölçekle (-255 ile +255 arası parlaklık, 0.0 ile 2.0 arası kontrast)
        brightness = int((brightness - 50) * 5.1)
        contrast = contrast / 50.0

        if brightness != 0 or contrast != 1:
            # ESKİ KOD (Sorunlu): image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
            
            # YENİ KOD (Düzeltilmiş):
            # addWeighted fonksiyonu 0'ın altına düşenleri otomatik 0 yapar, mutlak değer almaz.
            # Formül: resim * kontrast + 0 + parlaklık
            image = cv2.addWeighted(image, contrast, np.zeros_like(image), 0, brightness)
            
        return image

    def apply_grayscale(self, image, apply=True):
        if apply:
            if len(image.shape) == 2: return image
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def apply_inversion(self, image, apply=True):
        if apply: return cv2.bitwise_not(image)
        return image

    def apply_gaussian_blur(self, image, intensity=0):
        if intensity > 0:
            kernel_size = intensity * 2 + 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    def apply_sharpening(self, image, intensity=0):
        if intensity > 0:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            kernel[1, 1] += intensity * 0.5
            return cv2.filter2D(image, -1, kernel)
        return image

    def apply_portrait_mode(self, image, apply=True, blur_strength=5):
        if not apply or self.original_image is None: return image

        # Orijinal görüntüyü RGB'ye çevirip segmentasyon yapıyoruz.
        # Not: İşlenmiş görüntü (image) yerine orijinali kullanmak daha iyi sonuç verir ama filtrelerin etkisini maskelemek için 'image' üzerinde işlem yapacağız.

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb_image)

        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            if blur_strength > 0:
                kernel_size = blur_strength * 2 + 1
                blurred_background = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            else:
                blurred_background = image.copy()

            human_part = cv2.bitwise_and(image, mask_bgr)
            background_part = cv2.bitwise_and(blurred_background, cv2.bitwise_not(mask_bgr))
            return cv2.add(human_part, background_part)
        return image

    def apply_face_recognition_logic(self, image):

        if not self.face_rec_enabled or not self.known_face_encodings:
            return image

        # Hız optimizasyonu için görüntüyü küçült
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        # BGR -> RGB dönüşümü (face_recognition için)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Yüzleri bul
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Bilinmiyor"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Sonuçları çiz (Koordinatları tekrar 4 ile çarp)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Renk ayarı: Tanınan yeşil, bilinmeyen kırmızı
            color = (0, 255, 0) if name != "Bilinmiyor" else (0, 0, 255)

            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        return image

    def calculate_fps(self):
        current_time = time.time()
        fps = 1 / ((current_time - self.prev_time) + 1e-6)
        self.prev_time = current_time
        return fps

    def update_image(self):
        # Tüm işlemleri uygulayarak görüntüyü günceller

        if self.original_image is None:
            return

        # Orijinal görüntüden başla
        temp_image = self.original_image.copy()

        # 1. Temel Efektler Parlaklık, Kontrast, Blur, Keskinleştirme
        temp_image = self.apply_brightness_contrast(
            temp_image, self.brightness, self.contrast)

        temp_image = self.apply_gaussian_blur(
            temp_image, self.blur_intensity)

        temp_image = self.apply_sharpening(
            temp_image, self.sharpness)

        # 2. Portre Modu Gri tonlamadan ÖNCE yapılmalı!
        # Böylece renkli görüntü üzerinde maske oluşturulup uygulanır
        temp_image = self.apply_portrait_mode(
            temp_image, self.portrait_mode, self.portrait_blur)

        # 3. Renk Dönüşümleri Gri Tonlama ve Negatif
        if self.grayscale:
            temp_image = self.apply_grayscale(temp_image, True)
            # Eğer gri tonlamaysa, inversion için tekrar BGR'ye çevir
            if self.inversion:
                temp_image = self.apply_inversion(temp_image, True)
        else:
            # Gri tonlama yoksa, inversion'ı renkli uygula
            if self.inversion:
                temp_image = self.apply_inversion(temp_image, True)

        # 4. Görüntüyü tekrar 3 kanallı (Renkli) formata zorla
        # Yüz tanıma ve çizim işlemleri için gerekli
        if len(temp_image.shape) == 2:
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)

        # 5. Yüz Tanıma (En sonda ve görüntü kesinlikle 3 kanallıyken)
        if self.face_rec_enabled:
            temp_image = self.apply_face_recognition_logic(temp_image)

        self.processed_image = temp_image

        # FPS hesapla (webcam modunda)
        if self.webcam_mode:
            self.fps = self.calculate_fps()

    def draw_info(self, image):
        info_lines = []
        if self.webcam_mode:
            info_lines.append(f"Mod: WEBCAM (FPS: {self.fps:.1f})")
            info_lines.append("'c': Kare Yakala | 's': Kaydet")
        else:
            info_lines.append("Mod: DOSYA")
            info_lines.append("'s': Kaydet")

        info_lines.append("'r': Sifirla | 'q': Çikis")
        # Yüz Tanıma Durumu
        status = "AÇIK" if self.face_rec_enabled else "KAPALI"
        info_lines.append(f"Yuz Tanima: {status}")

        y_offset = 30
        for i, line in enumerate(info_lines):
            font_size = 0.6 if i >= 3 else 0.7
            thickness = 2 if i >= 3 else 2
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
            cv2.rectangle(image, (10, y_offset - 25), (10 + text_size[0] + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(image, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (0, 255, 0) if i < 3 else (255, 255, 255), thickness)
            y_offset += 30 if i < 2 else 25
        return image

    def create_trackbars(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 800)

        cv2.createTrackbar('Parlaklik', self.window_name, 50, 100, self.on_brightness_change)
        cv2.createTrackbar('Kontrast', self.window_name, 50, 100, self.on_contrast_change)
        cv2.createTrackbar('Bulaniklastirma', self.window_name, 0, 10, self.on_blur_change)
        cv2.createTrackbar('Keskinlestirme', self.window_name, 0, 10, self.on_sharpness_change)
        cv2.createTrackbar('Gri Tonlama', self.window_name, 0, 1, self.on_grayscale_change)
        cv2.createTrackbar('Negatif', self.window_name, 0, 1, self.on_inversion_change)
        cv2.createTrackbar('Portre Modu', self.window_name, 0, 1, self.on_portrait_change)
        cv2.createTrackbar('Portre Bulaniklik', self.window_name, 5, 15, self.on_portrait_blur_change)

        # --- YENİ TRACKBAR: Yüz Tanıma ---
        cv2.createTrackbar('Yuz Tanima', self.window_name, 0, 1, self.on_face_rec_change)

    # Trackbar callback fonksiyonları
    def on_brightness_change(self, val):
        self.brightness = val; self.update_image()

    def on_contrast_change(self, val):
        self.contrast = val; self.update_image()

    def on_blur_change(self, val):
        self.blur_intensity = val; self.update_image()

    def on_sharpness_change(self, val):
        self.sharpness = val; self.update_image()

    def on_grayscale_change(self, val):
        self.grayscale = val; self.update_image()

    def on_inversion_change(self, val):
        self.inversion = val; self.update_image()

    def on_portrait_change(self, val):
        self.portrait_mode = val; self.update_image()

    def on_portrait_blur_change(self, val):
        self.portrait_blur = val; self.update_image()

    # --- YENİ CALLBACK ---
    def on_face_rec_change(self, val):
        self.face_rec_enabled = val
        self.update_image()

    def reset_settings(self):
        # Sadece değişkenleri sıfırla (Görsel ayarları değil)
        self.brightness = 50
        self.contrast = 50
        self.blur_intensity = 0
        self.sharpness = 0
        self.grayscale = 0
        self.inversion = 0
        self.portrait_mode = 0
        self.portrait_blur = 5
        self.face_rec_enabled = 0

        # OpenCV Trackbar komutlarını (cv2.setTrackbarPos) SİLDİK.
        # Çünkü artık arayüzü PyQt yönetiyor.

        self.update_image()

    def select_mode(self):
        print("\n" + "=" * 50)
        print("MINI PHOTOSHOP + FACE ID")
        print("=" * 50)
        print("1. Webcam Kullan (Canlı)")
        print("2. Dosyadan Fotoğraf Yükle")
        print("3. Çıkış")
        while True:
            choice = input("\nSeçiminiz (1-3): ").strip()
            if choice == '1':
                return 'webcam'
            elif choice == '2':
                return 'file'
            elif choice == '3':
                return 'exit'
            else:
                print("Geçersiz seçim!")

    def select_file(self):
        file_path = input("\nDosya yolu (örn: foto.jpg): ").strip()
        if not os.path.exists(file_path):
            print(f"\n'{file_path}' bulunamadı! Mevcut dizindeki dosyalar:")
            files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for i, f in enumerate(files, 1): print(f"{i}. {f}")
            try:
                c = int(input(f"Seçim (1-{len(files)}): "))
                if 1 <= c <= len(files): return files[c - 1]
            except:
                pass
            return None
        return file_path

    def run(self):
        mode = self.select_mode()
        if mode == 'exit':
            return
        elif mode == 'webcam':
            if not self.start_webcam(): return
        elif mode == 'file':
            fp = self.select_file()
            if not fp or not self.load_image(fp): return

        self.create_trackbars()
        self.update_image()
        self.prev_time = time.time()

        print("\nKontroller: Trackbar'lar ile ayar yapin. 'q': Cikis, 'r': Sifirla")

        while True:
            if self.webcam_mode:
                if not self.get_next_webcam_frame(): break
                self.update_image()

            display_image = self.processed_image.copy()
            display_image = self.draw_info(display_image)
            cv2.imshow(self.window_name, display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_image(self.captured_image if (self.webcam_mode and self.captured_image is not None) else None)
            elif key == ord('r'):
                self.reset_settings()
            elif key == ord('c') and self.webcam_mode:
                self.capture_current_frame()

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MiniPhotoshop()
    app.run()