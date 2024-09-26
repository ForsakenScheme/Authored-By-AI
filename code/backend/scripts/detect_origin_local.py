import numpy as np

from joblib import load
from pathlib import Path
from sklearn.pipeline import Pipeline

from backend.utils.log import setup_logging

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QListView,
    QMessageBox,
    QDialog,
    QScrollArea,
    QApplication,
)
from PyQt5.QtCore import Qt

logger = setup_logging("local")


def predict_unknown_texts(model_pipeline: Pipeline, texts, min_length=50, web=False):
    """
    Predicts the labels for unknown texts using the custom pipeline.

    Parameters:
        model_pipeline (Pipeline): A trained pipeline to make predictions with.
        texts (list): A list of texts to make predictions on.
        min_length (int): The minimum length required for a text to be considered for prediction. Default is 50.

    Returns:
        list: A list of tuples containing the preidcted labels and probabilities for the input texts.
    """
    predictions = []
    logger.info("Predicting unknown texts...")
    for text in texts:
        logger.info("Predicting label for text:")
        logger.info(text)
        if len(text.strip().split()) < min_length:
            # If the text is too short, assign the label "short"
            predicted_label = np.str_("skipped (too short)")
            predicted_probability = np.array([-1, -1])
        else:
            # Otherwise, make predictions using the custom pipeline
            predicted_label = model_pipeline.predict([text])[0]
            # 0 is ai, 1 is human
            predicted_probability = model_pipeline.predict_proba([text])[0]
        predictions.append((predicted_label, predicted_probability))
    return predictions


class PredictionWindow(QDialog):
    """
    Dialog window to display predictions made by the origin detection model.

    Parameters:
        predictions (list): A list of tuples containing predicted labels and probabilities for each input text.

    Methods:
        closeEvent: Overrides the close event to close child windows.
        detect_origin: Function to detect the origin of the input text using a selected classifier.
    """

    def __init__(self, predictions):
        super().__init__()
        self.setWindowTitle("Predictions")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setBaseSize(600, 500)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(
            True
        )  # Allow the scroll area to resize its widget
        self.setLayout(QVBoxLayout())  # Set main layout

        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        for index, prediction in enumerate(predictions):
            result_index = index + 1
            label, probabilities = prediction

            # Determine the result message based on the label
            if label == "ai":
                ai_probability = probabilities[0]
                confidence = (
                    "≈ 100"
                    if ai_probability >= 0.9999
                    else f"{ai_probability * 100:.2f}"
                )
                result_message_text = (
                    f"<strong>Result paragraph {result_index}:</strong> Text appears to be written with the help of "
                    f"<strong><span style='color:red'>generative AI ({confidence}% confidence)</span></strong>."
                )
            elif label == "human":
                human_probability = probabilities[1]
                confidence = (
                    "≈ 100"
                    if human_probability >= 0.9999
                    else f"{human_probability * 100:.2f}"
                )
                result_message_text = (
                    f"<strong>Result paragraph {result_index}:</strong> Text appears to be written by a "
                    f"<strong><span style='color:green'>Human ({confidence}% confidence)</span></strong>."
                )
            elif label == "skipped (too short)":
                result_message_text = (
                    f"<strong>Result paragraph {result_index}:</strong> <span style='color:gray'>Skipped</span>. "
                    "The text has to be at least 50 words long for detection to be relevant."
                )
            else:
                result_message_text = f"<strong>Result paragraph {result_index}:</strong> An error occurred while analyzing the text."

            # Create QTextEdit to display result message
            result_text_edit = QTextEdit(result_message_text)
            result_text_edit.setReadOnly(True)
            result_text_edit.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )  # Allow text selection
            content_layout.addWidget(result_text_edit)

        # Set size policy for content widget
        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)


class BetterTextEdit(QTextEdit):
    """
    Subclass of QTextEdit with improved appearance and functionality.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(600)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setPlaceholderText("Enter the text here...")

        layout = QHBoxLayout(parent)
        layout.addWidget(self)
        layout.setAlignment(Qt.AlignCenter)


class DetectOriginWindow(QMainWindow):
    """
    Main window for the origin detection tool.

    Attributes:
        text_edit (BetterTextEdit): Text editor widget for inputting text.
        classifier_combo (QComboBox): Combo box for selecting the classifier model.
        child_windows (list): List to store child windows.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detect Origin")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.child_windows = []

        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(50, 50, 50, 50)  # Add margins

        # Description Label
        description_label = QLabel("Detection Tool")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        self.layout.addWidget(description_label)
        self.layout.addSpacing(30)

        # Text Description
        text_description = QLabel(
            "Input the text you want to detect the origin of. This can be any complete text or paragraph longer than 50 words."
        )
        text_description.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(text_description)
        self.layout.addSpacing(20)

        # Text Box
        self.text_edit = BetterTextEdit()
        self.layout.addWidget(self.text_edit)
        self.layout.addSpacing(20)

        # Classifier selection
        classifier_label = QLabel("Select Classifier:")
        self.layout.addWidget(classifier_label)
        self.classifier_combo = QComboBox()
        classifiers = [
            "Decision Tree",
            "Gradient Boosting",
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Random Forest",
            "Support Vector Machine",
            "Stacking Decision Tree",
            "Stacking Gradient Boosting",
            "Stacking Random Forest",
            "Stacking Support Vector Machine",
            "Stacking Logistic Regression",
            "Stacking Multinomial Naive Bayes",
            "Bagging Decision Tree",
            "Bagging Gradient Boosting",
            "Bagging Logistic Regression",
            "Bagging Multinomial Naive Bayes",
            "Bagging Random Forest",
            "Bagging Support Vector Machine",
        ]
        self.classifier_combo.addItems(classifiers)
        self.classifier_combo.setStyleSheet(
            """
        QComboBox QAbstractItemView::item {
            border-bottom: 1px solid lightgray;  /* Add separator */
            padding: 5px;
        }

        QComboBox QAbstractItemView::item:selected {
            background-color: #a1926b;  /* Highlight selected item */
        }
        """
        )
        # Set the view to a list view to make it scrollable
        self.classifier_combo.setView(QListView())
        self.layout.addWidget(self.classifier_combo)
        self.layout.addSpacing(20)

        # Button
        detect_button = QPushButton("Detect Origin")
        detect_button.clicked.connect(self.detect_origin)
        self.layout.addWidget(detect_button)

        # Add loading message
        self.layout.addSpacing(20)
        self.loading_label = QLabel(
            "Please be patient... The windows will be stuck for some time while ABAI computes the results."
        )
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.loading_label)
        self.loading_label.hide()

        # Set stretch to make the layout grow
        self.layout.addStretch()

        # Set size policy to make the layout expand
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def closeEvent(self, event):
        """
        Overrides the close event to close child windows.
        """
        for window in self.child_windows:
            window.close()

    def detect_origin(self):
        """
        Function to detect the origin of the input text using a selected classifier.
        """
        show_pred_window = True
        texts = self.text_edit.toPlainText()
        paragraphs = texts.split("\n")
        # Get rid of empty paragraphs
        paragraphs = [
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        ]
        if not paragraphs:
            QMessageBox.warning(
                self,
                "No Text",
                "Please input text into the text area before pressing the Detect Origin button.",
            )
            show_pred_window = False
            return
        selected_model = self.classifier_combo.currentText()
        # Construct the file path
        file_path = (
            Path(__file__).resolve().parent.parent.parent
            / "backend/models"
            / (selected_model + ".joblib")
        )
        try:
            # Load the model file
            model_pipeline = load(file_path)
            logger.info(f"Loaded model {selected_model} from file.")
        except FileNotFoundError:
            logger.info(f"Model file {selected_model} not found.")
            QMessageBox.warning(
                self,
                "Model not found",
                f"The model file for {selected_model} was not found. Please ensure it was trained and saved first.",
            )
            return
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while loading the model: {e}"
            )
            return
        self.loading_label.show()
        QApplication.processEvents()
        predictions = predict_unknown_texts(model_pipeline, paragraphs)
        for prediction in predictions:
            assert (
                prediction[0] == "ai"
                or prediction[0] == "human"
                or prediction[0] == "skipped (too short)"
            ), "Invalid prediction"
        if show_pred_window:
            # Hide the loading message
            self.loading_label.hide()
            prediction_window = PredictionWindow(predictions)
            self.child_windows.append(prediction_window)
            prediction_window.show()
            prediction_window.exec_()
