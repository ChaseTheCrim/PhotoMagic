import cv2
import numpy as np
import os
import face_recognition
from deepface import DeepFace

class AIProcessor:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        self.frame_counter = 0
        self.last_results = []
        
        self.gender_map = {"Woman": "Kadin", "Man": "Erkek"}
        self.emotion_map = {
            "happy": "Mutlu", "sad": "Uzgun", "angry": "Kizgin", 
            "surprise": "Saskin", "fear": "Korkmus", 
            "disgust": "Tiksinmis", "neutral": "Notr"
        }

    def reset_cache(self):
        # Yeni resim yüklendiğinde hafızayı sessizce sıfırla
        self.last_results = []
        self.frame_counter = 0

    def load_known_faces(self):
        folder = "FaceDatabase"
        if not os.path.exists(folder):
            os.makedirs(folder)
            return
        
        # Açılışta sadece burası bilgi versin
        print("AI Modülü: Yüz veritabanı taranıyor...")
        for filename in os.listdir(folder):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(folder, filename)
                try:
                    file_stream = np.fromfile(path, dtype=np.uint8)
                    img = cv2.imdecode(file_stream, cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)
                    if len(encodings) > 0:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        print(f" -> Öğrenildi: {name}")
                except: pass

    def process_frame(self, img, enable_face_rec=False, enable_age_gender=False):
        if not (enable_face_rec or enable_age_gender):
            return img

        self.frame_counter += 1
        
        # 1. YÜZ BULMA
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        
        # 2. ANALİZ KARARI
        face_count_changed = len(face_locations) != len(self.last_results)
        periodic_refresh = (self.frame_counter % 5 == 0)
        
        missing_feature = False
        if len(self.last_results) > 0:
            first = self.last_results[0]
            if enable_age_gender and not first.get("info"): missing_feature = True
            if enable_face_rec and not first.get("emotion"): missing_feature = True

        new_face_detected = (len(self.last_results) == 0 and len(face_locations) > 0)

        should_analyze = face_count_changed or periodic_refresh or missing_feature or new_face_detected

        if should_analyze:
            self.last_results = [] 
            
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for idx, (top, right, bottom, left) in enumerate(face_locations):
                loc_real = (top*4, right*4, bottom*4, left*4)
                
                # A. İsim Tanıma
                name = "Bilinmiyor"
                if self.known_face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[idx], tolerance=0.6)
                    dists = face_recognition.face_distance(self.known_face_encodings, face_encodings[idx])
                    if len(dists) > 0 and matches[np.argmin(dists)]:
                        name = self.known_face_names[np.argmin(dists)]

                # B. DeepFace Analizi
                emotion = ""
                age_gender = ""
                
                t, r, b, l = loc_real
                face_img = img[max(0, t):min(img.shape[0], b), max(0, l):min(img.shape[1], r)]

                if face_img.size > 0 and (enable_face_rec or enable_age_gender):
                    try:
                        actions = []
                        if enable_face_rec: actions.append('emotion')
                        if enable_age_gender: actions.extend(['age', 'gender'])
                        
                        if actions:
                            # Hata veren 'verbose' parametresi tamamen kaldırıldı
                            analysis = DeepFace.analyze(face_img, actions=actions, enforce_detection=False)
                            
                            if isinstance(analysis, list): analysis = analysis[0]

                            if enable_face_rec:
                                raw_emo = analysis.get('dominant_emotion')
                                emotion = self.emotion_map.get(raw_emo, raw_emo)
                            
                            if enable_age_gender:
                                age = analysis.get('age')
                                gen = analysis.get('dominant_gender')
                                gen_tr = self.gender_map.get(gen, gen)
                                age_gender = f"{gen_tr}, {age}"
                            
                    except:
                        pass # Hata olursa sessizce geç, program akışı bozulmasın

                self.last_results.append({
                    "loc": loc_real, "name": name, "emotion": emotion, "info": age_gender
                })

        # 3. ÇİZİM
        for res in self.last_results:
            top, right, bottom, left = res["loc"]
            
            color = (0, 255, 0)
            if res["emotion"] in ["Kizgin", "Korkmus", "Tiksinmis"]: color = (0, 0, 255)
            elif res["emotion"] == "Mutlu": color = (0, 255, 255)
            elif res["emotion"] == "Uzgun": color = (255, 0, 0)

            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            
            # Üst Etiket
            label_top = res["name"]
            if enable_face_rec and res["emotion"]: label_top += f" | {res['emotion']}"
            
            (w_top, h_top), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.rectangle(img, (left, top - 30), (left + w_top + 10, top), color, cv2.FILLED)
            cv2.putText(img, label_top, (left + 5, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 1)

            # Alt Etiket
            if enable_age_gender and res["info"]:
                label_bot = res["info"]
                # Dinamik font ölçeği (Resim boyutuna göre büyür/küçülür)
                font_scale = max(0.6, img.shape[1] / 1500.0) 
                
                (w_bot, h_bot), _ = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
                cv2.rectangle(img, (left, bottom), (left + w_bot + 10, bottom + 30), color, cv2.FILLED)
                cv2.putText(img, label_bot, (left + 5, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,0), 1)

        return img