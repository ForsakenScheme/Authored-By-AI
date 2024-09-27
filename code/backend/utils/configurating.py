import configparser
import sys
from backend.utils.log import setup_logging
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
)
from PyQt5.QtCore import Qt

logger = setup_logging("local")

config = configparser.ConfigParser(comment_prefixes="#", inline_comment_prefixes="#")
config.read("code/backend/config/config.ini")

max_size_int = sys.maxsize


class ConfigurationWindow(QMainWindow):
    """
    Configuration window for setting options.

    Attributes:
        options: Dictionary of options for the configuration window.
        config: Configuration settings for the window.
        configuration: Configuration settings for the window.

    Methods:
        save_config: Save the configuration settings.
        get_items_for_section: Get items for a specific section in the configuration window.
    """

    def __init__(self, localization_config):

        super().__init__()
        self.setWindowTitle("Configuration Menu")
        self.setBaseSize(850, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.localization_config = localization_config
        layout = QVBoxLayout(self.central_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(scroll_area)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        scroll_area.setWidget(scroll_widget)

        self.options = {}

        try:
            self.config = config
            self.configuration = {
                "Preprocessing": {
                    "punctuation": self.config.getboolean(
                        "Preprocessing", "punctuation"
                    ),
                    "stemm": self.config.getboolean("Preprocessing", "stemm"),
                    "lemmatize": self.config.getboolean("Preprocessing", "lemmatize"),
                },
                "TextWordCounter": {
                    "freqDist": self.config.getboolean("TextWordCounter", "freqDist"),
                    "bigrams": self.config.getboolean("TextWordCounter", "bigrams"),
                },
                "FeatureSelection": {
                    "nb_features": self.config.get("FeatureSelection", "nb_features")
                },
                "FeatureExtractionBeforePreprocessing": {
                    "stopWords": self.config.getboolean(
                        "FeatureExtractionBeforePreprocessing", "stopWords"
                    ),
                    "errorDetector": self.config.getboolean(
                        "FeatureExtractionBeforePreprocessing", "errorDetector"
                    ),
                    "punctuationFrequency": self.config.getboolean(
                        "FeatureExtractionBeforePreprocessing", "punctuationFrequency"
                    ),
                    "sentenceLength": self.config.getboolean(
                        "FeatureExtractionBeforePreprocessing", "sentenceLength"
                    ),
                    "namedEntity": self.config.getboolean(
                        "FeatureExtractionBeforePreprocessing", "namedEntity"
                    ),
                    "sentimentAnalysis": self.config.getboolean(
                        "FeatureExtractionBeforePreprocessing", "sentimentAnalysis"
                    ),
                },
                "FeatureExtractionAfterPreprocessing": {
                    "textWordCounter": self.config.getboolean(
                        "FeatureExtractionAfterPreprocessing", "textWordCounter"
                    ),
                    "wordLength": self.config.getboolean(
                        "FeatureExtractionAfterPreprocessing", "wordLength"
                    ),
                    "vocabularySize": self.config.getboolean(
                        "FeatureExtractionAfterPreprocessing", "vocabularySize"
                    ),
                },
                "CrossValidation": {
                    "nb_folds": self.config.get("CrossValidation", "nb_folds")
                },
                "RandomState": {
                    "random_state": self.config.get("RandomState", "random_state")
                },
            }
        except:
            logger.warning(
                "Configuration file not found. Creating a new one on configuration save."
            )
            open("code/backend/config/config.ini", "w").close()
            self.configuration = {
                "Preprocessing": {
                    "punctuation": False,
                    "stemm": False,
                    "lemmatize": False,
                },
                "TextWordCounter": {"freqDist": False, "bigrams": True},
                "FeatureSelection": {"nb_features": "all"},
                "FeatureExtractionBeforePreprocessing": {
                    "stopWords": True,
                    "errorDetector": True,
                    "punctuationFrequency": True,
                    "sentenceLength": True,
                    "namedEntity": True,
                    "sentimentAnalysis": True,
                },
                "FeatureExtractionAfterPreprocessing": {
                    "textWordCounter": True,
                    "wordLength": True,
                    "vocabularySize": True,
                },
                "CrossValidation": {"nb_folds": "10"},
                "RandomState": {"random_state": "42"},
            }

        sections = [
            ("Preprocessing", "Configure text cleaning options"),
            ("TextWordCounter", "Settings for the TextWordCounter"),
            ("FeatureSelection", "Set the number of features to select"),
            (
                "FeatureExtractionBeforePreprocessing",
                "Configure feature extraction options before preprocessing",
            ),
            (
                "FeatureExtractionAfterPreprocessing",
                "Configure feature extraction options after preprocessing",
            ),
            ("CrossValidation", "Select the number of folds for cross-validation"),
            ("RandomState", "Set the random state for reproducibility"),
        ]

        for section, description in sections:
            group_box = QGroupBox(section)
            group_layout = QVBoxLayout(group_box)
            scroll_layout.addWidget(group_box)
            group_layout.addWidget(QLabel(description))
            group_layout.addSpacing(10)
            for item, item_name in self.get_items_for_section(section):
                if item == 'Number of Features to Select (integer >= 1 or "all"):':
                    label = QLabel(item)
                    self.nb_features_entry = QLineEdit(
                        self.configuration["FeatureSelection"]["nb_features"]
                    )
                    group_layout.addWidget(label)
                    group_layout.addWidget(self.nb_features_entry)
                elif item == "Random State (integer >= 0):":
                    label = QLabel(item)
                    self.random_state_entry = QLineEdit(
                        self.configuration[section].get("random_state", "")
                    )
                    group_layout.addWidget(label)
                    group_layout.addWidget(self.random_state_entry)
                elif item == "Number of folds to perform (integer >= 2):":
                    label = QLabel(item)
                    self.nb_folds_entry = QLineEdit(
                        self.configuration[section].get("nb_folds", "")
                    )
                    group_layout.addWidget(label)
                    group_layout.addWidget(self.nb_folds_entry)
                else:
                    checkbox = QCheckBox(item)
                    checkbox.setChecked(self.configuration[section][item_name])
                    checkbox.setCursor(Qt.PointingHandCursor)
                    group_layout.addWidget(checkbox)
                    self.options[item_name] = checkbox
                group_layout.addSpacing(10)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_config)
        save_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(save_button)

    def save_config(self):
        """
        Save the configuration settings.
        """
        # Update the configuration for each section that has checkboxes
        for section_name, settings in self.configuration.items():
            for setting in settings.keys():
                if setting in ["nb_features", "nb_folds", "random_state"]:
                    continue
                checkbox = self.options[setting]
                self.config[section_name][setting] = str(checkbox.isChecked())

        # Make sure that stemming and lemmatization are not both selected
        if (
            self.config["Preprocessing"]["lemmatize"] == "True"
            and self.config["Preprocessing"]["stemm"] == "True"
        ):
            QMessageBox.warning(
                self,
                "Invalid input",
                "You cannot apply both stemming and lemmatization at the same time.",
            )
            return

        # Make sure that the number of features is a non-negative integer or "all"
        nb_features = self.nb_features_entry.text().strip()
        if (
            not (nb_features.isdigit() and int(nb_features) >= 0)
            and nb_features != "all"
        ):
            QMessageBox.warning(
                self,
                "Invalid input",
                'Number of features must be a non-negative integer or "all".',
            )
            return
        if int(nb_features) > max_size_int:
            QMessageBox.warning(
                self,
                f"Invalid input",
                "Number of features must be less than or equal to the maximum integer value. ({maxsize})",
            )
            return
        # Update the configuration for the number of features
        self.config["FeatureSelection"]["nb_features"] = nb_features

        # Make sure that the random state is a positive integer
        random_state = self.random_state_entry.text().strip()
        if not random_state.isdigit() or random_state == "":
            QMessageBox.warning(
                self, "Invalid input", "Random state must be an integer or None."
            )
            return
        if not int(random_state) >= 0:
            QMessageBox.warning(self, "Invalid input", "Random state must be >= 0.")
            return
        if int(random_state) > max_size_int:
            QMessageBox.warning(
                self,
                f"Invalid input",
                "Random state must be less than or equal to the maximum integer value. ({maxsize})",
            )
            return
        self.config["RandomState"]["random_state"] = random_state
        # Make sure that the number of folds is a positive integer >= 2
        nb_folds = self.nb_folds_entry.text().strip()
        if not nb_folds.isdigit():
            QMessageBox.warning(
                self, "Invalid input", "Number of folds must be an integer."
            )
            return
        if not int(nb_folds) >= 2:
            QMessageBox.warning(self, "Invalid input", "Number of folds must be >= 2.")
            return
        if int(nb_folds) > max_size_int:
            QMessageBox.warning(
                self,
                f"Invalid input",
                "Number of folds must be less than or equal to the maximum integer value. ({maxsize})",
            )
            return
        self.config["CrossValidation"]["nb_folds"] = nb_folds

        # Save the configuration
        with open("code/backend/config/config.ini", "w") as configfile:
            self.config.write(configfile)
        logger.info("Configuration saved successfully.")
        reply = QMessageBox.information(
            self, "Settings saved", "Settings have been saved successfully."
        )
        if reply == QMessageBox.Ok:
            self.close()

    def get_items_for_section(self, section):
        """
        Get items for a specific section in the configuration window.

        Parameters:
            section (str): The section name.

        Returns:
            list: A list of tuples containing item names and their corresponding keys.
        """
        items_map = {
            "Preprocessing": [
                ("Remove Punctuation", "punctuation"),
                ("Apply Stemming", "stemm"),
                ("Apply Lemmatization", "lemmatize"),
            ],
            "TextWordCounter": [
                ("Apply Frequency Distribution", "freqDist"),
                ("Apply Bigrams", "bigrams"),
            ],
            "FeatureSelection": [
                ('Number of Features to Select (integer >= 1 or "all"):', "nb_features")
            ],
            "FeatureExtractionBeforePreprocessing": [
                ("Apply Stop Words Ratio", "stopWords"),
                (
                    "Apply Error Detection (this increases ABAI's processing delay a lot.)",
                    "errorDetector",
                ),
                ("Apply Punctuation Frequency", "punctuationFrequency"),
                ("Apply Sentence Length", "sentenceLength"),
                ("Apply Named Entity Recognition", "namedEntity"),
                ("Apply Sentiment Analysis", "sentimentAnalysis"),
            ],
            "FeatureExtractionAfterPreprocessing": [
                ("Apply Text Word Counter", "textWordCounter"),
                ("Apply Word Length", "wordLength"),
                ("Apply Vocabulary Size", "vocabularySize"),
            ],
            "CrossValidation": [
                ("Number of folds to perform (integer >= 2):", "nb_folds")
            ],
            "RandomState": [("Random State (integer >= 0):", "random_state")],
        }
        return items_map.get(section, [])
