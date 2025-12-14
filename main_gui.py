import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.uic import loadUi
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

# Senin backend dosyanı çağırıyoruz
from yuz_tanima_mini import MiniPhotoshop

class PhotoshopApp(QMainWindow):
    def __init__(self):
        super(PhotoshopApp, self).__init__()
        
        # 1. Arayüzü Yükle
        try:
            loadUi('arayuz.ui', self)
        except Exception as e:
            print("HATA: 'arayuz.ui' dosyası yüklenemedi!")
            print(f"Detay: {e}")
            return

        # 2. Görüntü İşleme Motorunu Başlat
        self.processor = MiniPhotoshop()
        
        # --- DÜZELTME: Yüz Tanıma Manuel Olarak Kapatıldı ---
        self.processor.face_rec_enabled = 0 
        # ----------------------------------------------------

        self.mode = "webcam"
        if not self.processor.start_webcam():
            print("Webcam açılamadı, dosya moduna geçiliyor.")
            self.mode = "file"
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("Durum: Webcam Bulunamadı")
        else:
            if hasattr(self, 'lbl_status'):
                self.lbl_status.setText("Durum: Webcam Modu (Canlı)")

        # 3. AKILLI BAĞLANTILAR (Slider <-> SpinBox)
        # Eğer UI dosyasında bu sliderlar yoksa hata vermesin diye try-except blokları var
        if hasattr(self, 'slider_brightness'): self.setup_control(self.slider_brightness, self.spin_brightness, "brightness")
        if hasattr(self, 'slider_contrast'): self.setup_control(self.slider_contrast,   self.spin_contrast,   "contrast")
        if hasattr(self, 'slider_blur'): self.setup_control(self.slider_blur,       self.spin_blur,       "blur_intensity")
        if hasattr(self, 'slider_sharpness'): self.setup_control(self.slider_sharpness,  self.spin_sharpness,  "sharpness")
        
        # Portre Bulanıklığı
        if hasattr(self, 'slider_portrait_blur'):
            self.setup_control(self.slider_portrait_blur, self.spin_portrait_blur, "portrait_blur")

        # 4. Checkbox ve Butonlar
        
        # --- YÜZ TANIMA İPTAL EDİLDİ ---
        # self.chk_face_rec.stateChanged.connect(self.toggle_face_rec)
        
        # Portre Modu
        chk_portrait = getattr(self, 'chk_portrait_mode', None) or getattr(self, 'chk_portait_mode', None)
        if chk_portrait:
            chk_portrait.stateChanged.connect(self.toggle_portrait)

        # Efektler (Hata vermemesi için kontrol ederek bağlıyoruz)
        if hasattr(self, 'chk_grayscale'): self.chk_grayscale.stateChanged.connect(self.toggle_grayscale)
        if hasattr(self, 'chk_negative'): self.chk_negative.stateChanged.connect(self.toggle_negative)

        # --- DÜZELTME: Çökme Yaratan Butonlar Geçici Olarak Kapatıldı ---
        # Bu butonlar .ui dosyasında olmadığı için hata veriyordu.
        
        # if hasattr(self, 'btn_webcam'): self.btn_webcam.clicked.connect(self.switch_to_webcam)
        if hasattr(self, 'btn_photo'): self.btn_photo.clicked.connect(self.open_file_dialog)
        
        # Sadece btn_save ve btn_reset varsa bağla
        if hasattr(self, 'btn_save'): self.btn_save.clicked.connect(self.save_snapshot)
        if hasattr(self, 'btn_reset'): self.btn_reset.clicked.connect(self.reset_all)

        # 5. Timer (Görüntü Akışı)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) 

    def setup_control(self, slider, spinbox, param_name):
        try:
            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda val: self.set_processor_value(param_name, val))
            slider.setValue(spinbox.value())
        except AttributeError:
            pass

    def set_processor_value(self, param_name, value):
        if hasattr(self.processor, param_name):
            setattr(self.processor, param_name, value)
            self.processor.update_image()

    def toggle_face_rec(self):
        # Devre dışı
        pass

    def toggle_portrait(self):
        chk = getattr(self, 'chk_portrait_mode', None) or getattr(self, 'chk_portait_mode', None)
        if chk:
            self.processor.portrait_mode = 1 if chk.isChecked() else 0
            self.processor.update_image()

    def toggle_grayscale(self):
        self.processor.grayscale = 1 if self.chk_grayscale.isChecked() else 0
        self.processor.update_image()

    def toggle_negative(self):
        self.processor.inversion = 1 if self.chk_negative.isChecked() else 0
        self.processor.update_image()

    def switch_to_webcam(self):
        self.mode = "webcam"
        self.processor.webcam_mode = True
        self.processor.start_webcam()
        if not self.timer.isActive(): self.timer.start(30)
        if hasattr(self, 'lbl_status'): self.lbl_status.setText("Mod: WEBCAM (Canlı)")

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "", "Resimler (*.jpg *.jpeg *.png)")
        if file_name:
            if self.processor.load_image(file_name):
                self.mode = "file"
                if hasattr(self, 'lbl_status'): self.lbl_status.setText(f"Mod: DOSYA ({file_name.split('/')[-1]})")
                self.processor.update_image()

    def save_snapshot(self):
        self.processor.capture_current_frame()
        if self.processor.save_image():
            if hasattr(self, 'lbl_status'): self.lbl_status.setText("Durum: Fotoğraf Kaydedildi! ✅")
        else:
            if hasattr(self, 'lbl_status'): self.lbl_status.setText("Hata: Kaydedilemedi! ❌")

    def reset_all(self):
        self.processor.reset_settings()
        if hasattr(self, 'slider_brightness'): self.slider_brightness.setValue(50)
        if hasattr(self, 'slider_contrast'): self.slider_contrast.setValue(50)
        if hasattr(self, 'slider_blur'): self.slider_blur.setValue(0)
        if hasattr(self, 'slider_sharpness'): self.slider_sharpness.setValue(0)
        
        if hasattr(self, 'chk_face_rec'): self.chk_face_rec.setChecked(False)
        if hasattr(self, 'chk_grayscale'): self.chk_grayscale.setChecked(False)
        if hasattr(self, 'chk_negative'): self.chk_negative.setChecked(False)
        
        chk_portrait = getattr(self, 'chk_portrait_mode', None) or getattr(self, 'chk_portait_mode', None)
        if chk_portrait: chk_portrait.setChecked(False)
        
        if hasattr(self, 'lbl_status'): self.lbl_status.setText("Ayarlar Sıfırlandı.")

    def update_frame(self):
        if self.mode == "webcam":
            ret = self.processor.get_next_webcam_frame()
            if not ret: return

        self.processor.update_image()
        frame = self.processor.processed_image
        if frame is None: return

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # lbl_video kontrolü
        if hasattr(self, 'lbl_video'):
            lbl_w = self.lbl_video.width()
            lbl_h = self.lbl_video.height()
            scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_video.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        if self.processor.cap:
            self.processor.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoshopApp()
    window.show()
    sys.exit(app.exec())