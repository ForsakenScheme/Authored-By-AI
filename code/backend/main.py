import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.utils.log import setup_logging
from backend.utils.configurating import ConfigurationWindow
from backend.scripts.train_validate_test import TrainValidateTestWindow
from backend.scripts.detect_origin_local import DetectOriginWindow
from backend.scripts.database_functions import DatabaseWindow
from backend.scripts.suggestion import SuggestionWindow

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
)
from PyQt5.QtGui import QFont, QPixmap, QPainter, QIcon
from PyQt5.QtCore import Qt, QSize

import configparser

class BackgroundWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(
            self.rect(), QPixmap("code/backend/static/images/Background.png")
        )

class SettingsWindow(QDialog):
    def __init__(self, parent=None, localization_config=None):
        super().__init__(parent)
        self.localization_config = localization_config
        self.parent_window = parent
        self.setWindowTitle("Settings")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        # Language selection
        self.language_label = QLabel("Select data language:")
        self.language_combo = QComboBox()
        
        if self.localization_config:
            current_language = self.localization_config.get("Data", "language").capitalize()
        
        else: 
            # Default to English
            current_language = "English"

        # Populate combo box with available languages, placing the current language first
        available_languages = ["English", "French", "German"]

        # Add current language first
        self.language_combo.addItem(current_language)

        # Add other languages, avoiding duplicates
        for lang in available_languages:
            if lang != current_language:
                self.language_combo.addItem(lang)

        # Set current language as selected
        self.language_combo.setCurrentText(current_language)

        # Layout settings
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        # Confirm button
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.apply_language_change)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def apply_language_change(self):
        """Pass selected language to the parent window."""
        selected_language = self.language_combo.currentText().lower()
        
        if self.parent_window:
            self.parent_window.close_all_child_windows()
            self.parent_window.set_data_language(selected_language)
        
        # Save the configuration
        if self.localization_config:
            self.localization_config.set("Data", "language", selected_language)
            with open("code/backend/config/localization.ini", "w") as configfile:
                self.localization_config.write(configfile)
                
        QMessageBox.information(
            self, "Data language changed", "Language settings have been changed successfully."
        )
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABAI : Authored by AI")
        self.setMinimumSize(1600, 800)
        self.localization_config = configparser.ConfigParser(comment_prefixes="#", inline_comment_prefixes="#")
        self.localization_config.read("code/backend/config/localization.ini")
        self.data_language = self.localization_config.get("Data", "language")

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

        # Add cogwheel settings button in the top left corner
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(100, 10, 10, 10)
        # Load the cogwheel image as the button icon
        settings_button = QPushButton()
        settings_icon = QPixmap("code/backend/static/images/Cogwheel.png")
        scaled_icon = settings_icon.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        settings_button.setIcon(QIcon(scaled_icon))
        settings_button.setIconSize(QSize(80, 80))
        settings_button.setFixedSize(80, 80)
        settings_button.setStyleSheet("border: none;")
        settings_button.clicked.connect(self.open_settings_window)
        settings_button.setCursor(Qt.PointingHandCursor)
        top_bar_layout.addWidget(settings_button, alignment=Qt.AlignLeft)
        layout.addLayout(top_bar_layout)

        # Add main text title
        main_title_label = QLabel("ABAI: Authored by AI")
        main_title_label.setAlignment(Qt.AlignCenter)
        main_title_label.setStyleSheet(
            "font-size: 36pt; font-weight: bold; color: #333;" 
        )
        layout.addWidget(main_title_label)
        layout.addSpacing(50)

        # Add menu buttons
        menu_buttons = [
            ("Database", self.open_database_window),
            ("Configuration", self.open_configuration_window),
            ("Suggestions", self.open_suggestion_window),
            ("Train, Test and Validate", self.open_train_test_validate_window),
            ("Detect Origin", self.open_detect_origin_window),
        ]

        # Find the width of the longest button text
        max_width = max([len(text) for text, _ in menu_buttons])

        for button_text, handler in menu_buttons:
            button = QPushButton(button_text)
            button.setFixedWidth(max_width * 15)
            button.clicked.connect(handler)
            button.setStyleSheet(
                "QPushButton {"
                "   background-color:   #a1926b;"
                "   border: none;"
                "   color: #37342b;"
                "   padding: 20px 0;"  
                "   text-align: center;"
                "   border-radius: 8px;"
                "   font-size: 18pt;"  
                "} "
                "QPushButton:hover {"
                "   background-color:   #a1836b;"
                "}"
            )
            button.setCursor(Qt.PointingHandCursor)
            layout.addWidget(button, alignment=Qt.AlignHCenter)
            layout.addSpacing(10)
        layout.addSpacing(100)

        # Initialize window attributes
        self.database_window = None
        self.configuration_window = None
        self.train_validate_test_window = None
        self.detect_origin_window = None
        self.suggestion_window = None

    def set_data_language(self, language):
        """Set the data language locally."""
        self.data_language = language

    def open_settings_window(self):
        """Open the settings window to select language."""
        settings_window = SettingsWindow(parent=self, localization_config=self.localization_config)
        settings_window.exec_()

    def open_database_window(self):
        """Open the database window with the selected data language."""
        self.database_window = DatabaseWindow(localization_config=self.localization_config)
        self.database_window.show()

    def open_configuration_window(self):
        """Open the configuration window with the selected data language."""
        self.configuration_window = ConfigurationWindow(localization_config=self.localization_config)
        self.configuration_window.show()
        
    def open_suggestion_window(self):
        """Open the suggestion window with the selected data language."""
        self.suggestion_window = SuggestionWindow(localization_config=self.localization_config)
        self.suggestion_window.show()

    def open_train_test_validate_window(self):
        """Open the train, test, and validate window with the selected data language."""
        self.train_validate_test_window = TrainValidateTestWindow(localization_config=self.localization_config)
        self.train_validate_test_window.show()

    def open_detect_origin_window(self):
        """Open the detect origin window with the selected data language."""
        self.detect_origin_window = DetectOriginWindow(localization_config=self.localization_config)
        self.detect_origin_window.show()

    def closeEvent(self, event):
        """Display a message box for confirmation when the user tries to exit the application."""
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

    def close_all_child_windows(self):
        """Close all child windows when the language is changed."""
        if self.database_window:
            self.database_window.close()
        if self.configuration_window:
            self.configuration_window.close()
        if self.train_validate_test_window:
            self.train_validate_test_window.close()
        if self.detect_origin_window:
            self.detect_origin_window.close()
        if self.suggestion_window:
            self.suggestion_window.close()

def main():
    setup_logging()
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
