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
        self.mode = "webcam"
        if not self.processor.start_webcam():
            print("Webcam açılamadı, dosya moduna geçiliyor.")
            self.mode = "file"
            self.lbl_status.setText("Durum: Webcam Bulunamadı")
        else:
            self.lbl_status.setText("Durum: Webcam Modu (Canlı)")

        # 3. AKILLI BAĞLANTILAR (Slider <-> SpinBox)
        # Fonksiyon: setup_control(slider, spinbox, backend_degiskeni)
        
        # Temel Ayarlar (İsimleri düzelttiğini varsayıyorum)
        self.setup_control(self.slider_brightness, self.spin_brightness, "brightness")
        self.setup_control(self.slider_contrast,   self.spin_contrast,   "contrast")
        self.setup_control(self.slider_blur,       self.spin_blur,       "blur_intensity")
        self.setup_control(self.slider_sharpness,  self.spin_sharpness,  "sharpness")
        
        # AI Ayarları (Portre Bulanıklığı)
        # Eğer slider ismini 'slider_portrait_blur' yaptıysan:
        if hasattr(self, 'slider_portrait_blur'):
            self.setup_control(self.slider_portrait_blur, self.spin_portrait_blur, "portrait_blur")
        else:
            print("UYARI: 'slider_portrait_blur' bulunamadı. Eski isim (slider_AI_brightness) mi kaldı?")

        # 4. Checkbox ve Butonlar
        # Yüz Tanıma
        self.chk_face_rec.stateChanged.connect(self.toggle_face_rec)
        
        # Portre Modu (İsim hatası varsa diye iki ihtimali de deniyoruz)
        if hasattr(self, 'chk_portrait_mode'):
            self.chk_portrait_mode.stateChanged.connect(self.toggle_portrait)
        elif hasattr(self, 'chk_portait_mode'): # ui dosyasındaki olası yazım hatası
            self.chk_portait_mode.stateChanged.connect(self.toggle_portrait)

        # Efektler
        self.chk_grayscale.stateChanged.connect(self.toggle_grayscale)
        self.chk_negative.stateChanged.connect(self.toggle_negative)

        # Alt Butonlar
        self.btn_webcam.clicked.connect(self.switch_to_webcam)
        self.btn_file.clicked.connect(self.open_file_dialog)
        self.btn_save.clicked.connect(self.save_snapshot)
        self.btn_reset.clicked.connect(self.reset_all)

        # 5. Timer (Görüntü Akışı)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # 30ms ~ 33 FPS

    def setup_control(self, slider, spinbox, param_name):
        """Slider ve Spinbox'ı birbirine bağlar ve backend'i günceller"""
        try:
            # Slider -> Spinbox
            slider.valueChanged.connect(spinbox.setValue)
            # Spinbox -> Slider
            spinbox.valueChanged.connect(slider.setValue)
            # Her ikisi -> Backend Değişkeni
            slider.valueChanged.connect(lambda val: self.set_processor_value(param_name, val))
            # Başlangıç değerini eşitle
            slider.setValue(spinbox.value())
        except AttributeError:
            print(f"HATA: {param_name} için arayüz elemanları (Slider/Spinbox) bulunamadı!")

    def set_processor_value(self, param_name, value):
        if hasattr(self.processor, param_name):
            setattr(self.processor, param_name, value)
            self.processor.update_image()

    def toggle_face_rec(self):
        self.processor.face_rec_enabled = 1 if self.chk_face_rec.isChecked() else 0
        self.processor.update_image()

    def toggle_portrait(self):
        # Hangi checkbox isminin geçerli olduğunu bul
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
        self.lbl_status.setText("Mod: WEBCAM (Canlı)")
        self.lbl_status.setStyleSheet("color: #00ff00; font-weight: bold;")

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "", "Resimler (*.jpg *.jpeg *.png)")
        if file_name:
            if self.processor.load_image(file_name):
                self.mode = "file"
                self.lbl_status.setText(f"Mod: DOSYA ({file_name.split('/')[-1]})")
                self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
                self.processor.update_image()

    def save_snapshot(self):
        self.processor.capture_current_frame()
        if self.processor.save_image():
            self.lbl_status.setText("Durum: Fotoğraf Kaydedildi! ✅")
        else:
            self.lbl_status.setText("Hata: Kaydedilemedi! ❌")

    def reset_all(self):
        self.processor.reset_settings()
        # UI Elemanlarını da sıfırla (Sliderları 0 veya 50 yap)
        self.slider_brightness.setValue(50)
        self.slider_contrast.setValue(50)
        self.slider_blur.setValue(0)
        self.slider_sharpness.setValue(0)
        self.chk_face_rec.setChecked(False)
        self.chk_grayscale.setChecked(False)
        self.chk_negative.setChecked(False)
        
        # Portre checkbox sıfırlama
        if hasattr(self, 'chk_portrait_mode'): self.chk_portrait_mode.setChecked(False)
        if hasattr(self, 'chk_portait_mode'): self.chk_portait_mode.setChecked(False)
        
        self.lbl_status.setText("Ayarlar Sıfırlandı.")

    def update_frame(self):
        if self.mode == "webcam":
            ret = self.processor.get_next_webcam_frame()
            if not ret: return

        # Görüntü işleme tetikle (Dosya modunda slider oynatınca anlık görmek için)
        self.processor.update_image()
        
        frame = self.processor.processed_image
        if frame is None: return

        # OpenCV (BGR) -> Qt (RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Pencere boyutuna göre orantılı ölçekle (KeepAspectRatio)
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