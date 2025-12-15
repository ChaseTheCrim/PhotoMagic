import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

def main():
    # 1. Uygulama döngüsünü başlat
    app = QApplication(sys.argv)

    # 2. UI Dosyasını Yükle (PySide6 Yöntemi)
    loader = QUiLoader()
    file = QFile("qtGUI.ui")
    
    if not file.open(QFile.ReadOnly):
        print("HATA: 'qtGUI.ui' dosyası bulunamadı veya açılamadı!")
        sys.exit(-1)
        
    window = loader.load(file)
    file.close()

    if not window:
        print("HATA: UI dosyası yüklendi ama pencere oluşturulamadı.")
        sys.exit(-1)

    # 3. Pencereyi göster
    window.setWindowTitle("Skeleton Test")
    window.show()

    # 4. Programı açık tut
    sys.exit(app.exec())

if __name__ == "__main__":
    main()