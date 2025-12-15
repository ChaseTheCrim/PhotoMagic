import sys
import cv2
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QTimer, Qt, QFile
from PySide6.QtGui import QImage, QPixmap

# Backend kütüphanemiz
from mainlib import MiniPhotoshop

class PhotoshopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- UI YÜKLEME ---
        self.load_ui_file('qtGUI.ui')

        # 2. Motoru Başlat
        self.processor = MiniPhotoshop()
        self.mode = "file"
        
        # UI Elemanlarını Tanımla
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("Durum: Hazır (PySide6 Modu)")
            self.lbl_status.setStyleSheet("color: white;")

        # 3. BAĞLANTILAR
        if hasattr(self, 'slider_brightness'): self.setup_control(self.slider_brightness, self.spin_brightness, "brightness")
        if hasattr(self, 'slider_contrast'): self.setup_control(self.slider_contrast,   self.spin_contrast,   "contrast")
        if hasattr(self, 'slider_blur'): self.setup_control(self.slider_blur,       self.spin_blur,       "blur")
        if hasattr(self, 'slider_sharpness'): self.setup_control(self.slider_sharpness,  self.spin_sharpness,  "sharpen")
        if hasattr(self, 'slider_portrait_blur'): self.setup_control(self.slider_portrait_blur, self.spin_portrait_blur, "portrait_blur")

        # Checkboxlar
        if hasattr(self, 'chk_portrait_mode'): self.chk_portrait_mode.stateChanged.connect(self.toggle_portrait)
        if hasattr(self, 'chk_portait_mode'): self.chk_portait_mode.stateChanged.connect(self.toggle_portrait) 
        if hasattr(self, 'chk_grayscale'): self.chk_grayscale.stateChanged.connect(self.toggle_grayscale)
        if hasattr(self, 'chk_negative'): self.chk_negative.stateChanged.connect(self.toggle_negative)
        if hasattr(self, 'chk_canny'): self.chk_canny.stateChanged.connect(self.toggle_canny)
        # Phase 2: Yüz Tanıma Tuşu
        if hasattr(self, 'chk_face_rec'): self.chk_face_rec.stateChanged.connect(self.toggle_face_rec)

        # Butonlar
        if hasattr(self, 'btn_webcam'): self.btn_webcam.clicked.connect(self.toggle_webcam)
        
        if hasattr(self, 'btn_photo'): self.btn_photo.clicked.connect(self.open_file_dialog)
        elif hasattr(self, 'btn_file'): self.btn_file.clicked.connect(self.open_file_dialog)
            
        if hasattr(self, 'btn_save'): self.btn_save.clicked.connect(self.save_snapshot)
        if hasattr(self, 'btn_reset'): self.btn_reset.clicked.connect(self.reset_all)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_ui_file(self, filename):
        loader = QUiLoader()
        file = QFile(filename)
        if not file.open(QFile.ReadOnly):
            print(f"HATA: {filename} açılamadı!")
            sys.exit(-1)

        self.ui_window = loader.load(file) 
        file.close()
        
        if not self.ui_window:
            print("HATA: UI yüklenemedi!")
            sys.exit(-1)

        if isinstance(self.ui_window, QMainWindow):
            central_widget = self.ui_window.centralWidget()
            if central_widget:
                self.setCentralWidget(central_widget)
            else:
                print("UYARI: .ui dosyasında centralWidget bulunamadı!")
        else:
            self.setCentralWidget(self.ui_window)
            
        for widget in self.findChildren(QWidget):
            if widget.objectName():
                setattr(self, widget.objectName(), widget)
                
        # Pencere başlığını ayarla
        self.setWindowTitle("PhotoMagic")
        self.resize(1000, 700)

    def setup_control(self, slider, spinbox, param_name):
        try:
            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda val: self.set_processor_value(param_name, val))
            slider.setValue(spinbox.value())
        except AttributeError:
            pass

    def set_processor_value(self, param_name, value):
        if param_name in self.processor.settings:
            self.processor.settings[param_name] = value
            self.processor.update_image_pipeline()
            if self.mode == "file": self.update_frame()

    # --- TOGGLE FONKSİYONLARI ---
    def toggle_canny(self):
        val = 1 if self.chk_canny.isChecked() else 0
        self.processor.settings["canny_edge"] = val
        self.processor.update_image_pipeline()
        if self.mode == "file": self.update_frame()
        
    def toggle_face_rec(self):
        val = 1 if self.chk_face_rec.isChecked() else 0
        self.processor.settings["face_rec"] = val
        self.processor.update_image_pipeline()
        if self.mode == "file": self.update_frame()

    def toggle_portrait(self):
        chk = getattr(self, 'chk_portrait_mode', None) or getattr(self, 'chk_portait_mode', None)
        if chk:
            self.processor.settings["portrait_mode"] = 1 if chk.isChecked() else 0
            self.processor.update_image_pipeline()
            if self.mode == "file": self.update_frame()

    def toggle_grayscale(self):
        self.processor.settings["grayscale"] = 1 if self.chk_grayscale.isChecked() else 0
        self.processor.update_image_pipeline()
        if self.mode == "file": self.update_frame()

    def toggle_negative(self):
        self.processor.settings["negative"] = 1 if self.chk_negative.isChecked() else 0
        self.processor.update_image_pipeline()
        if self.mode == "file": self.update_frame()

    def toggle_webcam(self):
        if self.processor.webcam_active:
            # --- KAPATMA İŞLEMİ ---
            self.processor.stop_webcam()
            self.timer.stop()
            
            # UI Güncelle
            self.lbl_status.setText("Durum: Webcam Kapatıldı 🔒")
            self.lbl_status.setStyleSheet("color: red;")
            self.btn_webcam.setText("Webcam Aç")
            
            if hasattr(self, 'lbl_cam_status'):
                self.lbl_cam_status.setStyleSheet("""
                    background-color: red;
                    border-radius: 10px;
                    border: 1px solid gray;
                """)
        else:
            # --- AÇMA İŞLEMİ ---
            if self.processor.start_webcam():
                self.mode = "webcam"
                self.timer.start(30)
                
                self.lbl_status.setText("Durum: Webcam Canlı 🔴")
                self.lbl_status.setStyleSheet("color: #00ff00; font-weight: bold;")
                self.btn_webcam.setText("Webcam Kapat")
                
                if hasattr(self, 'lbl_cam_status'):
                    self.lbl_cam_status.setStyleSheet("""
                        background-color: #00ff00;
                        border-radius: 10px;
                        border: 1px solid gray;
                        box-shadow: 0px 0px 10px #00ff00;
                    """)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Fotoğraf Seç", "", "Resimler (*.jpg *.jpeg *.png)")
        if file_name:
            if self.processor.load_image(file_name):
                self.mode = "file"
                self.lbl_status.setText(f"Mod: DOSYA ({file_name.split('/')[-1]})")
                self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
                self.timer.stop() 
                self.update_frame()

    def save_snapshot(self):
        self.processor.capture_current_frame()
        if self.processor.save_image():
            self.lbl_status.setText("Durum: Kaydedildi (Yüksek Kalite) ✅")
        else:
            self.lbl_status.setText("Hata: Kaydedilemedi! ❌")

    def reset_all(self):
        widgets = [self.slider_brightness, self.slider_contrast, self.slider_blur, 
                   self.slider_sharpness, getattr(self, 'slider_portrait_blur', None)]
        checkboxes = [self.chk_grayscale, self.chk_negative, getattr(self, 'chk_canny', None),
                      getattr(self, 'chk_face_rec', None), getattr(self, 'chk_portrait_mode', None)]
        
        for w in widgets + checkboxes:
            if w: w.blockSignals(True)

        if hasattr(self, 'slider_brightness'): self.slider_brightness.setValue(50)
        if hasattr(self, 'slider_contrast'): self.slider_contrast.setValue(50)
        if hasattr(self, 'slider_blur'): self.slider_blur.setValue(0)
        if hasattr(self, 'slider_sharpness'): self.slider_sharpness.setValue(0)
        
        for chk in checkboxes:
            if chk: chk.setChecked(False)

        self.processor.reset_settings()

        for w in widgets + checkboxes:
            if w: w.blockSignals(False)
            
        if self.mode == "file": self.update_frame()
        self.lbl_status.setText("Ayarlar Sıfırlandı.")

    def update_frame(self):
        if self.mode == "webcam":
            ret = self.processor.get_next_frame()
            if not ret: return
            self.processor.update_image_pipeline()

        frame = self.processor.processed_image
        if frame is None: return

        # OpenCV (BGR) -> Qt (RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        if hasattr(self, 'lbl_video'):
            lbl_w = self.lbl_video.width()
            lbl_h = self.lbl_video.height()
            scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_video.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.processor.stop_webcam()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoshopApp()
    window.show()
    sys.exit(app.exec())