import cv2
import numpy as np
import mediapipe as mp
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# Yapay Zeka Modülünü Al
from AI_Core import AIProcessor

class MiniPhotoshop:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        self.ai_engine = AIProcessor() 

        self.full_res_image = None   
        self.proxy_image = None      
        self.processed_image = None  
        self.captured_image = None   

        self.settings = {
            "brightness": 50, "contrast": 50, "blur": 0, "sharpen": 0,
            "grayscale": 0, "negative": 0, "portrait_mode": 0, "portrait_blur": 5,
            "canny_edge": 0,
            "face_rec": 0,
            "age_gender": 0,
            "skeleton": 0
        }

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

    def load_image(self, image_path):
        try:
            file_stream = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(file_stream, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Hata: {e}")
            return False

        if image is None: return False
        self.stop_webcam()
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

    def process_image(self, img_input):
        if img_input is None: return None
        img = img_input.copy()

        # 1. Efektler
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

        if self.settings["portrait_mode"]:
            img = self.apply_portrait_mode(img, self.settings["portrait_blur"])

        if self.settings["grayscale"]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.settings["negative"]: img = cv2.bitwise_not(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.settings["negative"]:
            img = cv2.bitwise_not(img)
            
        # 2. Yapay Zeka İşlemleri
        if self.settings["face_rec"] or self.settings["age_gender"] or self.settings["skeleton"]:
            img = self.ai_engine.process_frame(
                img, 
                enable_face_rec=self.settings["face_rec"],
                enable_age_gender=self.settings["age_gender"],
                enable_skeleton=self.settings["skeleton"]
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

    def generate_unique_filename(self, directory, base_name, extension=".jpg"):
        counter = 1
        filename = f"{base_name}{extension}"
        full_path = os.path.join(directory, filename)
        while os.path.exists(full_path):
            filename = f"{base_name}_{counter}{extension}"
            full_path = os.path.join(directory, filename)
            counter += 1
        return full_path

    # Yüz iskeleti modu açıkken akıllı kayıt mekanizması
    def save_image_smart(self):
        s = self.settings
        is_skeleton_on = s.get("skeleton", 0)
        
        # Kirletici efekt kontrolü
        has_polluting_effects = (
            s.get("blur", 0) > 0 or s.get("sharpen", 0) > 0 or 
            s.get("grayscale", 0) == 1 or s.get("negative", 0) == 1 or 
            s.get("portrait_mode", 0) == 1 or
            s.get("face_rec", 0) == 1 or s.get("age_gender", 0) == 1
        )

        # Kontrol 1: Veritabanına Temiz Kayıt
        if is_skeleton_on and not has_polluting_effects:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(
                initialdir=self.ai_engine.db_dir,
                title="Yeni Kişi Olarak Kaydet (İsim Giriniz)",
                filetypes=[("JPG", "*.jpg")], defaultextension=".jpg"
            )
            root.destroy()

            if file_path:
                directory = os.path.dirname(file_path)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                final_path = self.generate_unique_filename(directory, base_name)
                
                # Temizlenmiş resmi kaydet
                image_to_save = self.full_res_image if self.full_res_image is not None else self.processed_image
                # Dosya modundaysa BGR'ye çevir (OpenCV standardı)
                # Not:Capture ettiysek zaten BGR'dir, o yüzden direkt kaydedilebilir.
                cv2.imwrite(final_path, image_to_save)
                
                self.ai_engine.load_known_faces() # Hafızayı yenile
                return "database", final_path

        # SENARYO B: Results'a Normal Kayıt
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"foto_{ts}.jpg"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, self.processed_image) # İşlenmişi kaydet
        return "results", path

    def reset_settings(self):
        for key in self.settings:
            self.settings[key] = 0
        self.settings["brightness"] = 50
        self.settings["contrast"] = 50
        self.settings["portrait_blur"] = 5
        self.update_image_pipeline()