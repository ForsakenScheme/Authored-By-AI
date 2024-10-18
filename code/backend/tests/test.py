from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QVBoxLayout

def main():
    app = QApplication([])  # Example QApplication
    dialog = QDialog()
    layout = QVBoxLayout()
    dialog.setLayout(layout)
    dialog.setWindowTitle("Test Dialog")
    dialog.exec_()

if __name__ == "__main__":
    main()