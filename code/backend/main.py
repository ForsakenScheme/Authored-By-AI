import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.utils.configurating import ConfigurationWindow
from backend.scripts.train_validate_test import TrainValidateTestWindow
from backend.scripts.detect_origin_local import DetectOriginWindow
from backend.scripts.database_functions import DatabaseWindow

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
)
from PyQt5.QtGui import QFont, QPixmap, QPainter
from PyQt5.QtCore import Qt


class BackgroundWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(
            self.rect(), QPixmap("code/backend/static/images/Background.png")
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABAI : Authored by AI")
        self.setFixedSize(1600, 800)
        central_widget = BackgroundWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Add picture in the top right corner
        picture_label = QLabel()
        pixmap = QPixmap("code/backend/static/images/ABAI_base.png")
        scaled_pixmap = pixmap.scaled(100, 100)
        picture_label.setPixmap(scaled_pixmap)
        picture_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        layout.addWidget(picture_label)

        # Add main text title
        main_title_label = QLabel("ABAI: Authored by AI")
        main_title_label.setAlignment(Qt.AlignCenter)
        main_title_label.setStyleSheet(
            "font-size: 24pt; font-weight: bold; color: #333;"
        )
        layout.addWidget(main_title_label)
        layout.addSpacing(50)

        # Add menu buttons
        menu_buttons = [
            ("Database", self.open_database_window),
            ("Configuration", self.open_configuration_window),
            ("Train, Test and Validate", self.open_train_test_validate_window),
            ("Detect Origin", self.open_detect_origin_window),
        ]

        # Find the width of the longest button text
        max_width = max([len(text) for text, _ in menu_buttons])

        for button_text, handler in menu_buttons:
            button = QPushButton(button_text)
            button.setFixedWidth(max_width * 15)  # Adjust multiplier as needed
            button.clicked.connect(handler)
            button.setStyleSheet(
                "QPushButton {"
                "   background-color:   #a1926b;"
                "   border: none;"
                "   color: #37342b;"
                "   padding: 15px 0;"
                "   text-align: center;"
                "   border-radius: 8px;"
                "}"
                "QPushButton:hover {"
                "   background-color:   #a1836b;"
                "}"
            )
            layout.addWidget(button, alignment=Qt.AlignHCenter)
            layout.addSpacing(10)  # Add spacing between buttons
        layout.addSpacing(100)  # Add extra space at the bottom

        # Initialize window attributes
        self.database_window = None
        self.configuration_window = None
        self.train_validate_test_window = None
        self.detect_origin_window = None

    # signal handler to close the TrainValidateTestWindow on error
    def closeTrainValidateTestWindow(self):
        if self.train_validate_test_window:
            self.train_validate_test_window.close()

    def open_database_window(self):
        self.database_window = DatabaseWindow()
        self.database_window.show()

    def open_configuration_window(self):
        self.configuration_window = ConfigurationWindow()
        self.configuration_window.show()

    def open_train_test_validate_window(self):
        self.train_validate_test_window = TrainValidateTestWindow()
        self.train_validate_test_window.show()
        self.train_validate_test_window.errorOccurred.connect(
            self.closeTrainValidateTestWindow
        )
        self.train_validate_test_window.setup_pipeline()

    def open_detect_origin_window(self):
        self.detect_origin_window = DetectOriginWindow()
        self.detect_origin_window.show()

    def closeEvent(self, event):
        # Display a message box for confirmation when the user tries to exit the application
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit? This will close all remaining windows.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            QApplication.quit()
        else:
            event.ignore()


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    screen = app.primaryScreen()
    dpi_x = screen.logicalDotsPerInchX()
    font_size = int(10 * dpi_x / 96)
    font = QFont("Roboto", font_size)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
