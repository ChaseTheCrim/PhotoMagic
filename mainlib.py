import cv2
import numpy as np
import mediapipe as mp
import os
import time

# MODÜLER YAPI: Yapay Zeka Uzmanını Çağırıyoruz
from AI_Core import AIProcessor

class MiniPhotoshop:
    def __init__(self):
        # --- MEDIAPIPE (Arka Plan Silme için) ---
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # --- YAPAY ZEKA MODÜLÜ ---
        self.ai_engine = AIProcessor() # AI işlerini bu arkadaş yapacak

        # --- GÖRÜNTÜ DEĞİŞKENLERİ ---
        self.full_res_image = None   
        self.proxy_image = None      
        self.processed_image = None  
        self.captured_image = None   

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
            "canny_edge": 0,
            "face_rec": 0,      # Modüler AI tetikleyicisi
            "age_gender": 0     # Modüler AI tetikleyicisi
        }

        # --- SİSTEM ---
        self.cap = None
        self.webcam_active = False 
        self.output_dir = "Results"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        self.prev_time = 0
        self.fps = 0

    def resize_image(self, image, width=None):
        if image is None: return None
        if width is None: return image
        (h, w) = image.shape[:2]
        if w <= width: return image
        r = width / float(w)
        dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # --- DOSYA YÖNETİMİ ---
    def load_image(self, image_path):
        try:
            file_stream = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(file_stream, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Hata: {e}")
            return False

        if image is None: return False
        self.stop_webcam()
        
        # [ÖNEMLİ] Yeni fotoğraf geldi, eski AI hafızasını sil!
        self.ai_engine.reset_cache()
        
        self.full_res_image = image
        self.proxy_image = self.resize_image(image, width=1024)
        self.update_image_pipeline()
        return True

    def start_webcam(self):
        if self.webcam_active: return True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return False
        self.webcam_active = True
        return True

    def stop_webcam(self):
        if self.cap: self.cap.release()
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

    # --- FİLTRE MOTORU (ANA PİPELİNE) ---
    def process_image(self, img_input):
        if img_input is None: return None
        img = img_input.copy()

        # 1. TEMEL EFEKTLER (Canny, Parlaklık vs.)
        if self.settings["canny_edge"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        b = int((self.settings["brightness"] - 50) * 5.1)
        c = self.settings["contrast"] / 50.0
        if b != 0 or c != 1:
            img = cv2.addWeighted(img, c, np.zeros_like(img), 0, b)

        if self.settings["blur"] > 0:
            k = self.settings["blur"] * 2 + 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        if self.settings["sharpen"] > 0:
            strength = self.settings["sharpen"] / 2.0
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            img = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

        # 2. PORTRE MODU
        if self.settings["portrait_mode"]:
            img = self.apply_portrait_mode(img, self.settings["portrait_blur"])

        # 3. RENK FİLTRELERİ
        if self.settings["grayscale"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.settings["negative"]: img = cv2.bitwise_not(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.settings["negative"]:
            img = cv2.bitwise_not(img)
            
        # 4. YAPAY ZEKA (MODÜLER ÇAĞRI) 🧠
        # Artık tüm karmaşık işlemler ai_core içinde yapılıyor.
        # Biz sadece "yap" diyoruz.
        if self.settings["face_rec"] or self.settings["age_gender"]:
            img = self.ai_engine.process_frame(
                img, 
                enable_face_rec=self.settings["face_rec"],
                enable_age_gender=self.settings["age_gender"]
            )

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
        self.processed_image = self.process_image(self.proxy_image)
        if self.webcam_active:
            cur_time = time.time()
            self.fps = 1 / ((cur_time - self.prev_time) + 1e-6)
            self.prev_time = cur_time

    def save_image(self):
        if not self.webcam_active and self.full_res_image is not None:
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
        for key in self.settings:
            self.settings[key] = 0
        self.settings["brightness"] = 50
        self.settings["contrast"] = 50
        self.settings["portrait_blur"] = 5
        self.update_image_pipeline()