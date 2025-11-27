import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

def main():
    # 1. Uygulama döngüsünü başlat
    app = QApplication(sys.argv)

    # 2. .ui dosyasını yükle
    ui_file_name = "arayuz.ui"
    ui_file = QFile(ui_file_name)
    
    if not ui_file.open(QFile.ReadOnly):
        print(f"Hata: {ui_file_name} dosyası açılamadı.")
        sys.exit(-1)

    loader = QUiLoader()
    window = loader.load(ui_file)
    ui_file.close()

    if not window:
        print(loader.errorString())
        sys.exit(-1)

    # 3. Pencereyi göster
    window.setWindowTitle("Bizim Proje")
    window.show()

    # 4. Programı açık tut
    sys.exit(app.exec())

if __name__ == "__main__":
    main()