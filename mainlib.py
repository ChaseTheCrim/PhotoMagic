import cv2
import numpy as np
import mediapipe as mp
import os
import time
# face_recognition şu an Phase 2 için bekletiliyor, import kalsın ama kullanmayacağız
import face_recognition

class MiniPhotoshop:
    def __init__(self):
        # --- MEDIAPIPE AYARLARI ---
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # --- GÖRÜNTÜ DEĞİŞKENLERİ (OPTIMIZASYON İÇİN) ---
        self.full_res_image = None   # Orijinal Yüksek Çözünürlüklü Hali
        self.proxy_image = None      # Küçültülmüş Kopya (Ekranda göstermek için)
        self.processed_image = None  # İşlenmiş Sonuç
        self.captured_image = None   # Snapshot

        # --- AYARLAR ---
        self.settings = {
            "brightness": 50,
            "contrast": 50,
            "blur": 0,
            "sharpen": 0,
            "grayscale": 0,
            "negative": 0,
            "portrait_mode": 0,
            "portrait_blur": 5,
            "canny_edge": 0,      # YENİ
            "face_rec": 0
        }

        # --- WEBCAM VE DURUM ---
        self.cap = None
        self.webcam_active = False # Başlangıçta kamera kapalı
        
        # --- ÇIKIS KLASÖRÜ ---
        self.output_dir = "islenmis_foto"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        self.prev_time = 0
        self.fps = 0

    # --- YARDIMCI: Görüntü Küçültme (Proxy) ---
    def resize_image(self, image, width=None):
        if image is None: return None
        if width is None: return image
        (h, w) = image.shape[:2]
        if w <= width: return image # Zaten küçükse dokunma
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # --- DOSYA YÖNETİMİ ---
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None: return False
        
        # Dosya yüklenince kamerayı güvenlik için kapat
        self.stop_webcam()
        
        self.full_res_image = image
        # Optimizasyon: Ekranda göstermek için max 1024px genişliğinde kopya
        self.proxy_image = self.resize_image(image, width=1024)
        self.update_image_pipeline()
        return True

    # --- WEBCAM AYARLARI ---
    def start_webcam(self):
        if self.webcam_active: return True
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("HATA: Kamera bulunamadı.")
            return False
            
        self.webcam_active = True
        return True

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
        self.webcam_active = False
        self.cap = None

    def get_next_frame(self):
        if not self.webcam_active or self.cap is None: return False
        
        ret, frame = self.cap.read()
        if ret:
            self.full_res_image = frame
            self.proxy_image = frame
            return True
        return False

    def capture_current_frame(self):
        if self.processed_image is not None:
            self.captured_image = self.processed_image.copy()
            return True
        return False

    # --- FİLTRE MOTORU (Pipeline) ---
    def process_image(self, img_input):
        if img_input is None: return None
        img = img_input.copy()

        # 1. CANNY EDGE DETECTION (Kenar Algılama)
        if self.settings["canny_edge"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 2. PARLAKLIK & KONTRAST
        b = int((self.settings["brightness"] - 50) * 5.1)
        c = self.settings["contrast"] / 50.0
        if b != 0 or c != 1:
            img = cv2.addWeighted(img, c, np.zeros_like(img), 0, b)

        # 3. BLUR (Bulanıklık)
        if self.settings["blur"] > 0:
            k = self.settings["blur"] * 2 + 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 4. SHARPENING 2.0 (Unsharp Masking)
        if self.settings["sharpen"] > 0:
            strength = self.settings["sharpen"] / 2.0
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            img = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

        # 5. PORTRE MODU
        if self.settings["portrait_mode"]:
            img = self.apply_portrait_mode(img, self.settings["portrait_blur"])

        # 6. RENK EFEKTLERİ
        if self.settings["grayscale"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.settings["negative"]: img = cv2.bitwise_not(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.settings["negative"]:
            img = cv2.bitwise_not(img)

        return img

    def apply_portrait_mode(self, image, blur_strength):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb)
        if results.segmentation_mask is not None:
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            k = blur_strength * 2 + 1
            blurred_bg = cv2.GaussianBlur(image, (k, k), 0)
            human = cv2.bitwise_and(image, mask_bgr)
            bg = cv2.bitwise_and(blurred_bg, cv2.bitwise_not(mask_bgr))
            return cv2.add(human, bg)
        return image

    def update_image_pipeline(self):
        """Ekrana basılacak (küçük) görüntüyü işle"""
        self.processed_image = self.process_image(self.proxy_image)
        if self.webcam_active:
            cur_time = time.time()
            self.fps = 1 / ((cur_time - self.prev_time) + 1e-6)
            self.prev_time = cur_time

    def save_image(self):
        """Kaydederken ORİJİNAL (Büyük) görüntüyü işleyip kaydeder"""
        # Dosya modundaysak büyük resmi işle, webcam ise ekrandakini
        if not self.webcam_active and self.full_res_image is not None:
            print("Yüksek çözünürlük işleniyor...")
            final_img = self.process_image(self.full_res_image)
        else:
            final_img = self.processed_image

        if final_img is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"foto_{ts}.jpg"
            path = os.path.join(self.output_dir, filename)
            cv2.imwrite(path, final_img)
            print(f"Kaydedildi: {path}")
            return True
        return False

    def reset_settings(self):
        # Ayarları varsayılana döndür
        for key in self.settings:
            self.settings[key] = 0
        self.settings["brightness"] = 50
        self.settings["contrast"] = 50
        self.settings["portrait_blur"] = 5
        self.update_image_pipeline()