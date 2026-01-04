import cv2
import numpy as np
import os
import face_recognition
import tensorflow as tf

#  Mod 2: DeepFace Kütüphanesi (Kullanmak isterseniz alttaki satırı açın)
# from deepface import DeepFace 

class AIProcessor:
    def __init__(self):
        # --- Ayarlar ---
        self.db_dir = "FaceDatabase"
        
        # =========================================================================
        # Mod 1: PhotoSynth Ekibinin Göz Nuru Modelleri
        # =========================================================================
        self.age_model_path = "models/age_model.h5"
        self.gen_emo_model_path = "models/gender_emotion_model.h5"
        
        # Modelleri Yükle
        self.age_model = self.load_model_safe(self.age_model_path, "Yaş")
        self.gen_emo_model = self.load_model_safe(self.gen_emo_model_path, "Cinsiyet/Duygu")
        
        # Etiketler (Custom Model İçin)
        self.emotion_labels = ['Kizgin', 'Igrenme', 'Korku', 'Mutlu', 'Uzgun', 'Saskin', 'Notr']
        # =========================================================================

        # =========================================================================
        # Mod 2: *Deepface Ayarları*
        # DeepFace için özel bir yüklemeye gerek yok, anlık çalışır.
        # Çeviri haritaları
        # =========================================================================
        # self.deepface_gender_map = {"Woman": "Kadin", "Man": "Erkek"}
        # self.deepface_emotion_map = {
        #     "happy": "Mutlu", "sad": "Uzgun", "angry": "Kizgin", 
        #     "surprise": "Saskin", "fear": "Korkmus", 
        #     "disgust": "Tiksinmis", "neutral": "Notr"
        # }
        # =========================================================================

        # Yüz Tanıma Hafızası (Her iki mod için ortak)
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Performans Yönetimi
        self.frame_counter = 0
        self.cached_results = [] 

    def load_model_safe(self, path, model_name):
        # Güvenlü yükleme
        if not os.path.exists(path):
            print(f"UYARI: {model_name} modeli bulunamadı -> {path}")
            return None
        
        print(f"{model_name} Modeli Yükleniyor...")
        try:
            model = tf.keras.models.load_model(path, compile=False)
            print(f"BAŞARI! {model_name} Modeli Hazır!")
            return model
        except Exception as e:
            print(f"HATA! {model_name} yüklenirken hata: {e}")
            return None

    def reset_cache(self):
        self.cached_results = []
        self.frame_counter = 0

    def load_known_faces(self):
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            return

        print(f"AI Core: {self.db_dir} taranıyor...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.db_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.db_dir, filename)
                try:
                    stream = open(path, "rb")
                    bytes = bytearray(stream.read())
                    numpyarray = np.asarray(bytes, dtype=np.uint8)
                    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
                    
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb)
                    if len(encs) > 0:
                        self.known_face_encodings.append(encs[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])
                        print(f" -> Öğrenildi: {os.path.splitext(filename)[0]}")
                except Exception as e:
                    print(f"Hata ({filename}): {e}")

    def predict_with_dual_models(self, face_img):
        """MOD 1: Bizim eğittiğimiz modellerle tahmin yapar"""
        if self.age_model is None or self.gen_emo_model is None:
            return "Model Eksik", ""

        try:
            # Preprocessing (224x224, 0-1 range)
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Yaş Tahmini
            age_preds = self.age_model.predict(img_batch, verbose=0)
            if isinstance(age_preds, list): age_val = age_preds[0][0][0]
            else: age_val = age_preds[0][0]
            age = int(age_val * 100)

            # Cinsiyet ve Duygu Tahmini
            ge_preds = self.gen_emo_model.predict(img_batch, verbose=0)
            
            # Çıktı sırası kontrolü
            out1 = ge_preds[0]
            out2 = ge_preds[1]
            
            if out1.shape[-1] == 7: # İlk çıktı 7 sınıflıysa Duygudur
                emotion_probs = out1[0]
                gender_val = out2[0][0]
            else: 
                gender_val = out1[0][0]
                emotion_probs = out2[0]

            gender = "Erkek" if gender_val > 0.5 else "Kadin"
            emotion_idx = np.argmax(emotion_probs)
            emotion = self.emotion_labels[emotion_idx]
            
            return emotion, f"{gender}, {age}"
            
        except Exception as e:
            print(f"Custom Model Hatası: {e}")
            return "", ""

    def process_frame(self, img, enable_face_rec=False, enable_age_gender=False, enable_skeleton=False):
        if not (enable_face_rec or enable_age_gender or enable_skeleton):
            return img

        self.frame_counter += 1
        
        # Yüz bulma hızlandırması(Resize)
        scale = 4
        small_frame = cv2.resize(img, (0, 0), fx=1/scale, fy=1/scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 1. YÜZLERİ BUL(Ortak Adım)
        face_locations = face_recognition.face_locations(rgb_small)
        
        # İskelet(Ortak Adım)
        face_landmarks_list = []
        if enable_skeleton:
            face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

        # Cache Kontrolü
        recalculate_deep = False
        if len(face_locations) != len(self.cached_results):
            recalculate_deep = True
        elif self.frame_counter % 5 == 0: 
            recalculate_deep = True
        
        if not recalculate_deep and len(self.cached_results) > 0:
            first = self.cached_results[0]
            if enable_age_gender and not first.get("info"): recalculate_deep = True
            if enable_face_rec and not first.get("emotion"): recalculate_deep = True

        # --- ANALİZ AŞAMASI ---
        if recalculate_deep:
            self.cached_results = [] 
            
            # Kimlik Tanıma (Mod 1 ve 2 Uyumlu)
            face_encodings = []
            if self.known_face_encodings:
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for idx, (top, right, bottom, left) in enumerate(face_locations):
                loc_real = (top*scale, right*scale, bottom*scale, left*scale)
                
                # İsim Bulma
                name = "Bilinmiyor"
                if len(face_encodings) > idx:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[idx], tolerance=0.6)
                    dists = face_recognition.face_distance(self.known_face_encodings, face_encodings[idx])
                    if len(dists) > 0:
                        best_idx = np.argmin(dists)
                        if matches[best_idx]: name = self.known_face_names[best_idx]

                emotion = ""
                age_gender = ""
                
                if enable_face_rec or enable_age_gender:
                    t, r, b, l = loc_real

                    # =========================================================
                    # Mod 1: *PhotoSynth Ekibi Modelleri*
                    # =========================================================
                    # Margin hesapla
                    h_face = b - t
                    w_face = r - l
                    margin = int(w_face * 0.15)
                    
                    t_m = max(0, t - margin)
                    b_m = min(img.shape[0], b + margin)
                    l_m = max(0, l - margin)
                    r_m = min(img.shape[1], r + margin)
                    
                    face_img = img[t_m:b_m, l_m:r_m]
                    
                    if face_img.size > 0:
                        emotion, age_gender = self.predict_with_dual_models(face_img)
                    # =========================================================

                    # =========================================================
                    # Mod 2: *Daha Kuvvetli DeepFace kütüphanesi*
                    # Kullanmak için üstteki bloğu kapatıp burayı açın.
                    # =========================================================
                    # # Margin olmadan direkt kesim (DeepFace kendi halleder)
                    # t_d = max(0, t); l_d = max(0, l)
                    # b_d = min(img.shape[0], b); r_d = min(img.shape[1], r)
                    # face_img_deep = img[t_d:b_d, l_d:r_d]
                    #
                    # if face_img_deep.size > 0:
                    #     try:
                    #         actions = []
                    #         if enable_face_rec: actions.append('emotion')
                    #         if enable_age_gender: actions.extend(['age', 'gender'])
                    #
                    #         if actions:
                    #             res = DeepFace.analyze(face_img_deep, actions=actions, enforce_detection=False, verbose=0)
                    #             if isinstance(res, list): res = res[0]
                    #
                    #             if enable_face_rec:
                    #                 raw_emo = res.get('dominant_emotion')
                    #                 emotion = self.deepface_emotion_map.get(raw_emo, raw_emo)
                    #
                    #             if enable_age_gender:
                    #                 d_age = res.get('age')
                    #                 d_gen = res.get('dominant_gender')
                    #                 d_gen_tr = self.deepface_gender_map.get(d_gen, d_gen)
                    #                 age_gender = f"{d_gen_tr}, {d_age}"
                    #     except: pass
                    # =========================================================

                    # Gereksiz veriyi temizle
                    if not enable_face_rec: emotion = ""
                    if not enable_age_gender: age_gender = ""
                
                self.cached_results.append({
                    "name": name, "emotion": emotion, "info": age_gender
                })

        # 3. Çizim Aşaması
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            top *= scale; right *= scale; bottom *= scale; left *= scale
            
            name = "Bilinmiyor"
            emotion = ""
            info = ""
            
            if idx < len(self.cached_results):
                res = self.cached_results[idx]
                name = res["name"]
                emotion = res["emotion"]
                info = res["info"]

            color = (0, 255, 0)
            if name == "Bilinmiyor": color = (0, 0, 255)

            # İskelet
            if enable_skeleton and idx < len(face_landmarks_list):
                landmarks = face_landmarks_list[idx]
                for feature, points in landmarks.items():
                    pts = np.array([(p[0]*scale, p[1]*scale) for p in points], np.int32)
                    is_closed = feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip']
                    cv2.polylines(img, [pts], is_closed, (255, 255, 0), 2)

            # Kutu ve Yazılar
            if enable_face_rec or enable_age_gender: 
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                
                label_top = name
                if enable_face_rec and emotion: label_top += f" | {emotion}"
                
                if label_top != "Bilinmiyor" or enable_face_rec:
                    (w, h), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                    cv2.rectangle(img, (left, top - 30), (left + w + 10, top), color, cv2.FILLED)
                    cv2.putText(img, label_top, (left + 5, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 1)
                
                if enable_age_gender and info:
                    font_scale = max(0.6, img.shape[1] / 1500.0)
                    (w_b, h_b), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
                    cv2.rectangle(img, (left, bottom), (left + w_b + 10, bottom + 30), color, cv2.FILLED)
                    cv2.putText(img, info, (left + 5, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,0), 1)

        return img