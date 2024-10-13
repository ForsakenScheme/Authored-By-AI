import matplotlib.pyplot as plt
import numpy as np
import configparser
import os

from sklearn.model_selection import StratifiedKFold, learning_curve
from backend.scripts.pipelines import UserConfigPipeline
from backend.scripts.database_functions import find_text_id_by_text
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils.log import get_logger
from time import time
from joblib import dump, load

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QHBoxLayout,
    QListView,
    QDialog,
    QMessageBox,
    QScrollArea,
    QWidget,
    QSizePolicy,
    QApplication,
    QVBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal

logger = get_logger(__name__)

def train_single(model: str, pipeline: UserConfigPipeline) -> float:
    """
    Train a single model based on the user's choice.

    Parameters:
        model (str): The name of the model to train.
        pipeline (UserConfigPipeline): The pipeline object used for training.

    Returns:
        train_time (float): The time taken to train the model in seconds.
    """
    pipeline.classifier_name = model
    start = time()

    # Train the model
    pipeline.train()
    train_time = time() - start

    return train_time

def grid_search_best_of_all_models(pipeline: UserConfigPipeline) -> tuple[dict, float, str, float]:
    """
    Perform a grid search to find the best model based on the user's config choice.

    Parameters:
        pipeline (UserConfigPipeline): The pipeline object used for grid search.

    Returns:
        grid_search_time (float): The time taken to perform the grid search in seconds.
        
    """
    start = time()

    # Perform grid_search_best_of_all
    best_parameters, best_score, best_classifier = pipeline.grid_search_best_of_all()
    grid_search_time = time() - start
    
    return best_parameters, best_score, best_classifier, grid_search_time
    

def grid_search_single(model: str, pipeline: UserConfigPipeline) -> float:
    """
    Perform a grid search for a single model based on the user's choice.

    Parameters:
        model (str): The name of the model to perform a grid search on.
        pipeline (UserConfigPipeline): The pipeline object used for grid search.
        param_grid (dict): The parameter grid to search over.
        scoring (str): The scoring metric to optimize.
        cv (int): The number of cross-validation folds.

    Returns:
        grid_search_time (float): The time taken to perform the grid search in seconds.
        best_parameters_for_score (list): A list of tuples containing the best parameters and scores for each scoring metric.
    """
    pipeline.classifier_name = model
    start = time()

    # Perform grid search
    best_parameters_for_score = pipeline.grid_search()
    grid_search_time = time() - start

    return grid_search_time, best_parameters_for_score


def validate_single(model, pipeline: UserConfigPipeline, data_language: str) -> tuple[float, dict]:
    """
    Validate a single model based on the user's choice.

    Parameters:
        model (str): The name of the model to validate.
        pipeline (UserConfigPipeline): The pipeline object used for validation.

    Returns:
        validation_time(float): The time taken to validate the model in seconds.
        validation_dict(dictionary): The validation results dictionary.
    """
    pipeline.classifier_name = model
    start = time()

    # Validate the model
    validation_dict = pipeline.validate()
    validation_time = time() - start
    # Store validation metrics in a joblib file along the existing model files
    filtered_metrics = {k: v for k, v in validation_dict.items() if k != "cm"}
    dump(filtered_metrics, os.path.join(os.getcwd(), f"code/backend/models/{data_language}", model + " Metrics" + ".joblib"))
    logger.info(f"Validation metrics for model {model} saved in {os.path.join(os.getcwd(), f'code/backend/models/{data_language}', model + " Metrics" + '.joblib')}.")

    return validation_time, validation_dict


def test_single(model, pipeline: UserConfigPipeline):
    """
    Test a single model based on the user's choice.

    Parameters:
        model (str): The name of the model to test.
        pipeline (UserConfigPipeline): The pipeline object used for testing.

    Returns:
        test_time (float): The time taken to test the model in seconds.
        nb_test_texts (int): Total number of texts tested.
        nb_test_ai_texts (int): Number of AI texts tested.
        nb_test_human_texts (int): Number of human texts tested.
        nb_test_ai_predictions (int): Number of AI predictions.
        nb_test_human_predictions (int): Number of human predictions.
        nb_test_skips (int): Number of skipped texts due to insufficient number of words.
        nb_test_correct_predictions (int): Number of correct predictions.
        test_dict (dict): Test results dictionary.
        predictions (List[str]): List of predictions.
        predict_probas (List[List[float]]): List of prediction probabilities.
    """
    pipeline.classifier_name = model
    # remove random state to get a test set that is not always the same to simulate unseen data
    pipeline.random_state = None
    start = time()

    # Test the model
    (
        nb_test_ai_texts,
        nb_test_human_texts,
        nb_test_ai_predictions,
        nb_test_human_predictions,
        nb_test_skips,
        nb_test_correct_predictions,
        test_dict,
        predictions,
        predict_probas,
    ) = pipeline.test()
    test_time = time() - start
    nb_test_texts = len(predictions)

    return (
        test_time,
        nb_test_texts,
        nb_test_ai_texts,
        nb_test_human_texts,
        nb_test_ai_predictions,
        nb_test_human_predictions,
        nb_test_skips,
        nb_test_correct_predictions,
        test_dict,
        predictions,
        predict_probas,
    )


def train_validate_test_single(model, pipeline: UserConfigPipeline, data_language: str) -> tuple:
    """
    Train, validate, and test a single model based on the user's choice.

    Parameters:
        model (str): The name of the model to train, validate, and test.
        pipeline (UserConfigPipeline): The pipeline object used for training, validation, and testing.

    Returns:
        train_time (float): The time taken to train the model in seconds.
        validation_time (float): The time taken to validate the model in seconds.
        test_time (float): The time taken to test the model in seconds.
        train_validate_test_time (float): The total time taken for training, validation, and testing in seconds.
        nb_test_texts (int): Total number of texts tested.
        nb_test_ai_texts (int): Number of AI texts tested.
        nb_test_human_texts (int): Number of human texts tested.
        nb_test_ai_predictions (int): Number of AI predictions.
        nb_test_human_predictions (int): Number of human predictions.
        nb_test_skips (int): Number of skipped texts due to insufficient number of words.
        nb_test_correct_predictions (int): Number of correct predictions.
        validation_dict (dict): Validation results dictionary.
        test_dict (dict): Test results dictionary.
        predictions (List[str]): List of predictions.
        predict_probas (List[List[float]]): List of prediction probabilities.
    """
    pipeline.classifier_name = model
    logger.info(f"Training, validating, and testing model {pipeline.classifier_name}.")
    start = time()

    # Train the model
    train_time = train_single(model, pipeline)

    # Validate the model
    logger.info(f"Loading the just trained model {pipeline.classifier_name}.")
    pipeline.loadCustomPipeline(model)
    logger.info(f"Successfully loaded model {pipeline.classifier_name}.")
    validation_time, validation_dict = validate_single(model, pipeline, data_language)

    # Test the model
    (
        test_time,
        nb_test_texts,
        nb_test_ai_texts,
        nb_test_human_texts,
        nb_test_ai_predictions,
        nb_test_human_predictions,
        nb_test_skips,
        nb_test_correct_predictions,
        test_dict,
        predictions,
        predict_probas,
    ) = test_single(model, pipeline)

    train_validate_test_time = time() - start
    logger.info(
        f"Finished training, validating, and testing model {pipeline.classifier_name} in {train_validate_test_time:.2f} seconds.\n"
    )

    return (
        train_time,
        validation_time,
        test_time,
        train_validate_test_time,
        nb_test_texts,
        nb_test_ai_texts,
        nb_test_human_texts,
        nb_test_ai_predictions,
        nb_test_human_predictions,
        nb_test_skips,
        nb_test_correct_predictions,
        validation_dict,
        test_dict,
        predictions,
        predict_probas,
    )


class TrainResultWindow(QDialog):
    """
    A QDialog window to display the training results for selected models.

    Parameters:
        selected_models (List[str]): List of selected models.
        dict_train_time (Dict[str, float]): Dictionary containing training times for each model.
    """

    def __init__(
        self,
        selected_models,
        dict_train_time,
    ):
        super().__init__()
        self.setWindowTitle("Train results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(
            True
        )  # Allow the scroll area to resize its widget

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)

        # Add content to content widget
        content = ""
        for model in selected_models:
            train_time = dict_train_time[model]
            content += f"<pre><b>Training results for {model}</b>\n\n"
            content += f"\tTrained in {train_time:.2f} seconds.\n\n"
            content += "</pre>\n\n==================================================================================\n\n"

        # Create QLabel widget to display the content
        content_label = QLabel(content)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(content_label)

        # Set size policy for content widget
        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)


class MatplotlibWindow(QWidget):
    """
    A PyQt5 window for plotting learning curves using Matplotlib.

    Attributes:
        figure (matplotlib.figure.Figure): The Matplotlib figure used for plotting.
        canvas (matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg): The Matplotlib canvas for displaying the figure.

    Parameters:
        parent (QWidget, optional): The parent widget. Defaults to None.

    Methods:
    plot_learning_curve: Plot the learning curve for a given model using Matplotlib.
    """

    def __init__(self, parent=None):
        super(MatplotlibWindow, self).__init__(parent)
        self.setWindowTitle("Learning curve")
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_learning_curve(self, pipeline, X_train, y_train):
        logger.info(
            f"Plotting learning curve for model {pipeline.named_steps['classification']}."
        )
        logger.info(f"Number of samples : {len(X_train)}")
        train_scores, valid_scores = learning_curve(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        )[-2:]
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)
        train_sizes_abs = (np.linspace(0.1, 1.0, 10) * len(X_train)).astype(int)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.fill_between(
            train_sizes_abs,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            train_sizes_abs,
            valid_scores_mean - valid_scores_std,
            valid_scores_mean + valid_scores_std,
            alpha=0.1,
            color="orange",
        )
        ax.plot(
            train_sizes_abs,
            train_scores_mean,
            "o-",
            color="blue",
            label="Training score",
        )
        ax.plot(
            train_sizes_abs,
            valid_scores_mean,
            "o-",
            color="orange",
            label="Validation score",
        )
        ax.set_xlabel("Training samples", fontsize=16)
        ax.set_ylabel("Score", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(loc="best", fontsize=14)
        ax.set_title("Learning Curve", fontsize=18)
        ax.set_ylim([0.0, 1.0])
        self.canvas.draw()
        logger.info(
            f"Finished plotting learning curve for model {pipeline.named_steps['classification']}."
        )


class GridSearchWindow(QDialog):
    """
    A QDialog window to display the results of a grid search.

    Parameters:
        selected_models (List[str]): The selected models.
        dict_grid_search_time (Dict[str, float]): The time taken to perform the grid search for each model.
        dict_grid_search_lists (Dict[str, List[Tuple[Tuple[str, float], Dict[str, Any]]]): The grid search results for each model.
    """

    def __init__(
        self,
        selected_models,
        dict_grid_search_time,
        dict_grid_search_lists,
    ):
        super().__init__()
        self.setWindowTitle("Validation results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(
            True
        )  # Allow the scroll area to resize its widget

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)

        # Add content to content widget
        # example [(("precision", 0.88)), {"classification__C": 1, "classification__kernel": "linear"}),]
        content = ""
        for model in selected_models:
            grid_search_time = dict_grid_search_time[model]
            content += f"<pre><b>Grid search results for {model}</b>\n\n"
            content += (
                f"\tGrid search  performed in {grid_search_time:.2f} seconds.\n\n"
            )

            grid_search_list = dict_grid_search_lists[model]
            for metric in grid_search_list:
                metric_name = metric[0][0]
                metric_score = metric[0][1]
                best_params_for_metric_dict = metric[1]
                content += f"Best achievable score for <b>{metric_name}</b>: {metric_score * 100:.2f}%\n"
                content += f"Recommended parameter configuration to achieve this:\n\n"
                for parameter, value in best_params_for_metric_dict.items():
                    content += f"\t{parameter}: {value}\n"
                content += "\n"
            content += "</pre>\n\n==================================================================================\n\n"

        # Create QLabel widget to display the content
        content_label = QLabel(content)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(content_label)

        # Set size policy for content widget
        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)

class ValidationResultWindow(QDialog):
    """
    A QDialog window to display the validation results for selected models.

    Parameters:
        selected_models (List[str]): List of selected models.
        dict_validation_time (Dict[str, float]): Dictionary containing validation times for each model.
        dict_validation_dicts (Dict[str, dict]): Dictionary containing validation results for each model.
    """
    def __init__(
        self,
        selected_models = None,
        dict_validation_time = None,
        dict_validation_dicts = None,
    ):
        super().__init__()
        self.setWindowTitle("Validation results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(
            True
        )  # Allow the scroll area to resize its widget

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)

        # Add content to content widget
        content = ""
        for model in selected_models:
            if dict_validation_time is not None:
                validation_time = dict_validation_time[model]
            validation_dict = dict_validation_dicts[model]
            content += f"<pre><b>Validation results for {model}</b>\n\n"
            if dict_validation_time is not None and validation_time is not None:
                content += f"\tValidated in {validation_time:.2f} seconds.\n\n"
            content += f"<b>Validation metrics:</b>\n\n\tAccuracy: {validation_dict['accuracy'] * 100:.2f}%\n\tPrecision: {validation_dict['precision'] * 100:.2f}%\n\tRecall: {validation_dict['recall'] * 100:.2f}%\n\tF1 Score: {validation_dict['f1'] * 100:.2f}%"
            if 'cm' in validation_dict:
                content += f"\n\tConfusion matrix:\n{validation_dict['cm']}\n\n"
            content += "</pre>\n\n==================================================================================\n\n"

        # Create QLabel widget to display the content
        content_label = QLabel(content)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(content_label)

        # Set size policy for content widget
        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)


class TestResultWindow(QDialog):
    """
    A QDialog window to display the test results for selected models.

    Parameters:
        dict_pipelines (Dict[str, UserConfigPipeline]): Dictionary containing pipelines for each model.
        dict_test_times (Dict[str, float]): Dictionary containing test times for each model.
        dict_nb_test_texts (Dict[str, int]): Dictionary containing total number of texts tested for each model.
        dict_nb_test_ai_texts (Dict[str, int]): Dictionary containing number of AI texts tested for each model.
        dict_nb_test_human_texts (Dict[str, int]): Dictionary containing number of human texts tested for each model.
        dict_nb_test_ai_predictions (Dict[str, int]): Dictionary containing number of AI predictions for each model.
        dict_nb_test_human_predictions (Dict[str, int]): Dictionary containing number of human predictions for each model.
        dict_nb_test_skips (Dict[str, int]): Dictionary containing number of skipped texts for each model.
        dict_nb_test_correct_predictions (Dict[str, int]): Dictionary containing number of correct predictions for each model.
        dict_test_dicts (Dict[str, dict]): Dictionary containing test results for each model.
        dict_predictions (Dict[str, List[str]]): Dictionary containing predictions for each model.
        dict_predict_probas (Dict[str, List[List[float]]]): Dictionary containing prediction probabilities for each model.
    """

    def __init__(
        self,
        dict_pipelines,
        dict_test_times,
        dict_nb_test_texts,
        dict_nb_test_ai_texts,
        dict_nb_test_human_texts,
        dict_nb_test_ai_predictions,
        dict_nb_test_human_predictions,
        dict_nb_test_skips,
        dict_nb_test_correct_predictions,
        dict_test_dicts,
        dict_predictions,
        dict_predict_probas,
        data_language,
    ):
        super().__init__()
        self.setWindowTitle("Test results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(
            True
        )  # Allow the scroll area to resize its widget

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)

        # Add content to content widget
        content = ""
        for model in dict_pipelines.keys():
            pipeline = dict_pipelines[model]
            test_time = dict_test_times[model]
            nb_test_texts = dict_nb_test_texts[model]
            nb_test_ai_texts = dict_nb_test_ai_texts[model]
            nb_test_human_texts = dict_nb_test_human_texts[model]
            nb_test_ai_predictions = dict_nb_test_ai_predictions[model]
            nb_test_human_predictions = dict_nb_test_human_predictions[model]
            nb_test_skips = dict_nb_test_skips[model]
            nb_test_correct_predictions = dict_nb_test_correct_predictions[model]
            test_dict = dict_test_dicts[model]
            predictions = dict_predictions[model]
            predict_probas = dict_predict_probas[model]

            content += f"<pre><b>Test results for {model}</b>\n\n"
            content += f"\tTested in {test_time:.2f} seconds.\n\n"
            content += f"\tTested on a total of {nb_test_texts} texts of which {nb_test_ai_texts} where AI texts and {nb_test_human_texts} human texts. \n\tA total of {nb_test_skips} texts were skipped because of insuffiscient number of words.\n\tAfter keeping only acceptable length texts, {nb_test_correct_predictions} out of {nb_test_texts - nb_test_skips} correct predictions were made.\n\tFor these texts, {nb_test_ai_predictions} were predicted as AI and {nb_test_human_predictions} as human.\n\n"
            content += f"<b>Test metrics:</b>\n\n\tAccuracy: {test_dict['accuracy'] * 100:.2f}%\n\tPrecision: {test_dict['precision'] * 100:.2f}%\n\tRecall: {test_dict['recall'] * 100:.2f}%\n\tF1 Score: {test_dict['f1'] * 100:.2f}%\n\tConfusion matrix:\n{test_dict['cm']}\n\n"
            content += "<b>Test results:</b>\n\n"
            i = 0
            for text, prediction, predict_proba in zip(
                pipeline.X_test, predictions, predict_probas
            ):  
                if prediction == "ai":
                    ai_confidence = predict_proba[0] * 100
                    if pipeline.y_test[i] == "ai":
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text, data_language=data_language)}): predicted {prediction}({ai_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='green'>Correct !</font>\n"
                    else:
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text, data_language=data_language)}): predicted {prediction}({ai_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='red'>False !</font>\n"
                elif prediction == "human":
                    human_confidence = predict_proba[1] * 100
                    if pipeline.y_test[i] == "human":
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text, data_language=data_language)}): predicted {prediction}({human_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='green'>Correct !</font>\n"
                    else:
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text, data_language=data_language)}): predicted {prediction}({human_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='red'>False !</font>\n"
                else:
                    content += f"\tText {i+1} (ID {find_text_id_by_text(text, data_language=data_language)}): skipped (too short).\n"
                i += 1
            content += "</pre>\n\n==================================================================================\n\n"

        # Create QLabel widget to display the content
        content_label = QLabel(content)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(content_label)

        # Set size policy for content widget
        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)


class TrainValidateTestResultWindow(QDialog):
    """
    A QDialog window to display the results of training, validation, and testing for multiple models.

    Parameters:
        dict_pipelines (dict): A dictionary mapping model names to pipeline objects.
        dict_train_time (dict): A dictionary mapping model names to the time taken for training each model.
        dict_validation_time (dict): A dictionary mapping model names to the time taken for validation each model.
        dict_test_times (dict): A dictionary mapping model names to the time taken for testing each model.
        dict_train_validation_test_times (dict): A dictionary mapping model names to the total time taken for training, validation, and testing each model.
        dict_nb_test_texts (dict): A dictionary mapping model names to the total number of texts tested for each model.
        dict_nb_test_ai_texts (dict): A dictionary mapping model names to the number of AI texts tested for each model.
        dict_nb_test_human_texts (dict): A dictionary mapping model names to the number of human texts tested for each model.
        dict_nb_test_ai_predictions (dict): A dictionary mapping model names to the number of AI predictions for each model.
        dict_nb_test_human_predictions (dict): A dictionary mapping model names to the number of human predictions for each model.
        dict_nb_test_skips (dict): A dictionary mapping model names to the number of skipped texts for each model.
        dict_nb_test_correct_predictions (dict): A dictionary mapping model names to the number of correct predictions for each model.
        dict_validation_dicts (dict): A dictionary mapping model names to the validation results dictionary for each model.
        dict_test_dicts (dict): A dictionary mapping model names to the test results dictionary for each model.
        dict_predictions (dict): A dictionary mapping model names to the list of predictions for each model.
        dict_predict_probas (dict): A dictionary mapping model names to the list of prediction probabilities for each model.
    """

    def __init__(
        self,
        dict_pipelines,
        dict_train_time,
        dict_validation_time,
        dict_test_times,
        dict_train_validation_test_times,
        dict_nb_test_texts,
        dict_nb_test_ai_texts,
        dict_nb_test_human_texts,
        dict_nb_test_ai_predictions,
        dict_nb_test_human_predictions,
        dict_nb_test_skips,
        dict_nb_test_correct_predictions,
        dict_validation_dicts,
        dict_test_dicts,
        dict_predictions,
        dict_predict_probas,
    ):
        super().__init__()
        self.setWindowTitle("Train, validate, test results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(
            True
        )  # Allow the scroll area to resize its widget

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)

        # Add content to content widget
        content = ""
        for model in dict_pipelines.keys():
            pipeline = dict_pipelines[model]
            train_time = dict_train_time[model]
            validation_time = dict_validation_time[model]
            test_time = dict_test_times[model]
            train_validate_test_time = dict_train_validation_test_times[model]
            nb_test_texts = dict_nb_test_texts[model]
            nb_test_ai_texts = dict_nb_test_ai_texts[model]
            nb_test_human_texts = dict_nb_test_human_texts[model]
            nb_test_ai_predictions = dict_nb_test_ai_predictions[model]
            nb_test_human_predictions = dict_nb_test_human_predictions[model]
            nb_test_skips = dict_nb_test_skips[model]
            nb_test_correct_predictions = dict_nb_test_correct_predictions[model]
            validation_dict = dict_validation_dicts[model]
            test_dict = dict_test_dicts[model]
            predictions = dict_predictions[model]
            predict_probas = dict_predict_probas[model]

            content += f"<pre><b>Train, validate and test results for {model}</b>\n\n"
            content += f"\tTrained in {train_time:.2f} seconds. \n\tValidated in {validation_time:.2f} seconds. \n\tTested in {test_time:.2f} seconds. \n\tTrained, validated and tested in a total of {train_validate_test_time:.2f}\n\n"
            content += f"\tTested on a total of {nb_test_texts} texts of which {nb_test_ai_texts} where AI texts and {nb_test_human_texts} human texts. \n\tA total of {nb_test_skips} texts were skipped because of insuffiscient number of words.\n\tAfter keeping only acceptable length texts, {nb_test_correct_predictions} out of {nb_test_texts - nb_test_skips} correct predictions were made.\n\tFor these texts, {nb_test_ai_predictions} were predicted as AI and {nb_test_human_predictions} as human.\n\n"
            content += f"<b>Validation metrics:</b>\n\n\tAccuracy: {validation_dict['accuracy'] * 100:.2f}%\n\tPrecision: {validation_dict['precision'] * 100:.2f}%\n\tRecall: {validation_dict['recall'] * 100:.2f}%\n\tF1 Score: {validation_dict['f1'] * 100:.2f}%\n\tConfusion matrix:\n{validation_dict['cm']}\n\n"
            content += f"<b>Test metrics:</b>\n\n\tAccuracy: {test_dict['accuracy'] * 100:.2f}%\n\tPrecision: {test_dict['precision'] * 100:.2f}%\n\tRecall: {test_dict['recall'] * 100:.2f}%\n\tF1 Score: {test_dict['f1'] * 100:.2f}%\n\tConfusion matrix:\n{test_dict['cm']}\n\n"
            content += "<b>Test results:</b>\n\n"
            i = 0
            for text, prediction, predict_proba in zip(
                pipeline.X_test, predictions, predict_probas
            ):
                if prediction == "ai":
                    ai_confidence = predict_proba[0] * 100
                    if pipeline.y_test[i] == "ai":
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text)}): predicted {prediction}({ai_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='green'>Correct !</font>\n"
                    else:
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text)}): predicted {prediction}({ai_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='red'>False !</font>\n"
                elif prediction == "human":
                    human_confidence = predict_proba[1] * 100
                    if pipeline.y_test[i] == "human":
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text)}): predicted {prediction}({human_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='green'>Correct !</font>\n"
                    else:
                        content += f"\tText {i+1} (ID {find_text_id_by_text(text)}): predicted {prediction}({human_confidence:.2f} %) - actual {pipeline.y_test[i]}. <font color='red'>False !</font>\n"
                else:
                    content += f"\tText {i+1} (ID {find_text_id_by_text(text)}): skipped (too short).\n"
                i += 1
            content += "</pre>\n\n==================================================================================\n\n"

        # Create QLabel widget to display the content
        content_label = QLabel(content)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(content_label)

        # Set size policy for content widget
        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)


class TrainValidateTestWindow(QMainWindow):
    """
    A QMainWindow for training, validating, and testing selected models.

    Attributes:
        train_result_window: A window to display training results.
        validation_result_window: A window to display validation results.
        test_result_window: A window to display test results.
        train_validate_test_result_window: A window to display results of training, validation, and testing.
        pipeline: The machine learning pipeline.

    Methods:
        setup_pipeline: Set up the machine learning pipeline.
        check_config: Check the pipeline configuration for consistency.
        train_models: Train the selected models.
        validate_models: Validate the selected models.
        test_models: Test the selected models.
        train_validate_test_models: Train, validate, and test the selected models.
        closeEvent: Close the windows associated with the TrainValidateTestWindow.
        start_action: Start the selected action.
    """

    errorOccurred = pyqtSignal(str)

    def __init__(self, localization_config: configparser.ConfigParser):
        super().__init__()
        self.setWindowTitle("Train, validate and test")
        self.setGeometry(200, 200, 1200, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.childWindows = []
        self.childPlots = []
        self.localization_config = localization_config
        self.data_language = self.localization_config.get("Data", "language")

        # Empty windows for initialization
        self.train_result_window = None
        self.validation_result_window = None
        self.test_result_window = None
        self.train_validate_test_result_window = None

        # Initialize pipeline
        self.pipeline = None
        self.setup_pipeline()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(100, 100, 100, 100)

        # Title
        title_label = QLabel("Train, validate and test the selected model(s)")
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: #333;")
        self.layout.addWidget(title_label)
        self.layout.addSpacing(50)

        # Description
        description_label = QLabel(
            "Training and validation will be based on the settings from the config.ini file.\n\nSelect the action you want to perform in the dropbox and tick the model(s) you want to apply it to:"
        )
        description_label.setStyleSheet("color: #666;")
        self.layout.addWidget(description_label)
        self.layout.addSpacing(30)

        # Action selection dropdown
        self.action_combo = QComboBox()
        self.action_combo.setStyleSheet(
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
        self.action_combo.setView(QListView())
        self.action_combo.view().setCursor(Qt.PointingHandCursor)
        self.actions = [
            "Train (using the configuration file)",
            "Learning curve (using the configuration file)",
            "Grid search (using the configuration file)",
            "Validate (loading existing model)",
            "Get validation metrics from saved metrics file",
            "Test (loading existing model)",
            "Train, validate and test (using the configuration file)",
        ]
        self.action_combo.addItems(self.actions)
        self.action_combo.setCursor(Qt.PointingHandCursor)
        self.layout.addWidget(self.action_combo)
        self.layout.addSpacing(30)

        # Model selection checkboxes
        models = [
            "Decision Tree",
            "Gradient Boosting",
            "Logistic Regression (L1)",
            "Logistic Regression (L2)",
            "Multinomial Naive Bayes",
            "Random Forest",
            "Support Vector Machine",
            "Stacking Decision Tree",
            "Stacking Gradient Boosting",
            "Stacking Random Forest",
            "Stacking Support Vector Machine",
            "Stacking Logistic Regression (L1)",
            "Stacking Logistic Regression (L2)",
            "Stacking Multinomial Naive Bayes",
            "Bagging Decision Tree",
            "Bagging Gradient Boosting",
            "Bagging Logistic Regression (L1)",
            "Bagging Logistic Regression (L2)",
            "Bagging Multinomial Naive Bayes",
            "Bagging Random Forest",
            "Bagging Support Vector Machine",
        ]

        models_layout = QHBoxLayout()
        models_layout.addStretch()
        # Create three columns for the checkboxes
        num_models = len(models)
        num_per_column = (
            num_models + 2
        ) // 3  # Add 2 to ensure enough rows for uneven division
        for i in range(3):
            column_layout = QVBoxLayout()
            column_layout.setSpacing(20)
            start_index = i * num_per_column
            end_index = min((i + 1) * num_per_column, num_models)
            for model in models[start_index:end_index]:
                checkbox = QCheckBox(model)
                checkbox.setCursor(Qt.PointingHandCursor)
                column_layout.addWidget(checkbox)
            models_layout.addStretch()
            models_layout.addLayout(column_layout)
        models_layout.addStretch()
        self.layout.addLayout(models_layout)
        self.layout.addSpacing(50)

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(
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
        self.start_button.clicked.connect(self.start_action)
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.layout.addWidget(self.start_button)
        self.layout.addSpacing(25)

        # Add loading message
        self.loading_label = QLabel(
            "Please be patient... The windows will be stuck for some time while ABAI computes the task."
        )
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.loading_label)
        self.loading_label.hide()

    def setup_pipeline(self):
        """
        Set up the machine learning pipeline.
        """
        try:
            self.pipeline = UserConfigPipeline()
        except Exception as e:
            logger.error(f"Error while initializing pipeline: {e}")
            self.errorOccurred.emit(str(e))
            QMessageBox.critical(
                self,
                "Error",
                'Error while initializing pipeline.\n\nProbably a bad setup of the database. You can check the logs in the "code/backend/logs/" folder.',
            )

    def check_config(self):
        """
        Check the pipeline configuration for consistency.
        """
        if self.pipeline is None:
            logger.info(
                "Pipeline config cannot be verified because it is not initialized."
            )
            QMessageBox.critical(
                self,
                "Error",
                "Pipeline config cannot be verified because it is not initialized.",
            )
            raise Exception(
                "Pipeline config cannot be verified because it is not initialized."
            )
        if (
            self.pipeline.config.get("Preprocessing", "stemm") == "True"
            and self.pipeline.config.get("Preprocessing", "lemmatize") == "True"
        ):
            logger.error(
                "Stemming and lemmatization cannot be enabled at the same time."
            )
            QMessageBox.critical(
                self,
                "Error",
                "Stemming and lemmatization cannot be enabled at the same time.",
            )
            raise Exception(
                "Stemming and lemmatization cannot be enabled at the same time."
            )
        if (
            not self.pipeline.config.get("FeatureSelection", "nb_features").isdigit()
            and self.pipeline.config.get("FeatureSelection", "nb_features") != "all"
        ):
            logger.error("Number of features must be a positive integer or 'all'.")
            QMessageBox.critical(
                self, "Error", "Number of features must be a positive integer or 'all'."
            )
            raise Exception("Number of features must be a positive integer or 'all'.")
        if (
            self.pipeline.config.get("FeatureSelection", "nb_features") <= "0"
            and self.pipeline.config.get("FeatureSelection", "nb_features") != "all"
        ):
            logger.error("Number of features must be a positive integer or 'all'.")
            QMessageBox.critical(
                self, "Error", "Number of features must be a positive integer or 'all'."
            )
            raise Exception("Number of features must be a positive integer or 'all'.")

    def train_models(self, selected_models):
        """
        Train the selected models and show the results in a new window.

        Parameters:
            selected_models (list): A list of selected models.
        """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        self.check_config
        dict_train_time = {}

        for model in selected_models:
            self.pipeline.setClassifierName(model)
            self.pipeline.setCustomPipeline()
            train_time = train_single(model, self.pipeline)

            # Store results in dictionaries
            dict_train_time[model] = train_time

        # Display results for each models in a new window
        train_result_window = TrainResultWindow(
            selected_models,
            dict_train_time,
        )
        self.childWindows.append(train_result_window)
        train_result_window.show()

    def grid_search_models(self, selected_models):
        """ """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        dict_grid_search_time = {}
        dict_grid_search_lists = {}
        
        for model in selected_models:
            self.pipeline.setClassifierName(model)
            self.pipeline.setParamGrid(model)
            self.pipeline.setCustomPipeline()
            grid_search_time, grid_search_list = grid_search_single(
                model, self.pipeline
            )

            # Store results in dictionaries
            dict_grid_search_time[model] = grid_search_time
            dict_grid_search_lists[model] = grid_search_list

        # Display results for each models in a new window
        grid_search_window = GridSearchWindow(
            selected_models,
            dict_grid_search_time,
            dict_grid_search_lists,
        )
        self.childWindows.append(grid_search_window)
        grid_search_window.show()

    def plot_learning_curve_models(self, selected_models):
        """
        Plot the learning curve for the selected model.

        Parameters:
            selected_models (str): The selected models.
        """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        for model in selected_models:
            self.pipeline.setClassifierName(model)
            self.pipeline.setCustomPipeline()
            learning_curve_widget = MatplotlibWindow()
            learning_curve_widget.plot_learning_curve(
                self.pipeline.custom_pipeline,
                self.pipeline.X_train,
                self.pipeline.y_train,
            )
            self.childPlots.append(learning_curve_widget)
            learning_curve_widget.show()

    def validate_models(self, selected_models):
        """
        Validate the selected models and show the results in a new window.

        Parameters:
            selected_models (list): A list of selected models.
        """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        dict_validation_time = {}
        dict_validation_dicts = {}

        for model in selected_models:
            self.pipeline.setClassifierName(model)
            self.pipeline.loadCustomPipeline(model)
            validation_time, validation_dict = validate_single(model, self.pipeline, self.data_language)
            # Display results for each models in a new window
            # Store results in dictionaries
            dict_validation_time[model] = validation_time
            dict_validation_dicts[model] = validation_dict

        # Display results for each models in a new window
        validation_result_window = ValidationResultWindow(
            selected_models=selected_models,
            dict_validation_time=dict_validation_time,
            dict_validation_dicts=dict_validation_dicts,
        )
        self.childWindows.append(validation_result_window)
        validation_result_window.show()

    def test_models(self, selected_models):
        """
        Test the selected models and show the results in a new window.

        Parameters:
            selected_models (list): A list of selected models.
        """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        dict_pipelines = {}
        dict_test_times = {}
        dict_nb_test_texts = {}
        dict_nb_test_ai_texts = {}
        dict_nb_test_human_texts = {}
        dict_nb_test_ai_predictions = {}
        dict_nb_test_human_predictions = {}
        dict_nb_test_skips = {}
        dict_nb_test_correct_predictions = {}
        dict_test_dicts = {}
        dict_predictions = {}
        dict_predict_probas = {}
        for model in selected_models:
            self.pipeline.setClassifierName(model)
            self.pipeline.loadCustomPipeline(model)
            (
                test_time,
                nb_test_texts,
                nb_test_ai_texts,
                nb_test_human_texts,
                nb_test_ai_predictions,
                nb_test_human_predictions,
                nb_test_skips,
                nb_test_correct_predictions,
                test_dict,
                predictions,
                predict_probas,
            ) = test_single(model, self.pipeline)
            # Store results in dictionaries
            dict_pipelines[model] = self.pipeline
            dict_test_times[model] = test_time
            dict_nb_test_texts[model] = nb_test_texts
            dict_nb_test_ai_texts[model] = nb_test_ai_texts
            dict_nb_test_human_texts[model] = nb_test_human_texts
            dict_nb_test_ai_predictions[model] = nb_test_ai_predictions
            dict_nb_test_human_predictions[model] = nb_test_human_predictions
            dict_nb_test_skips[model] = nb_test_skips
            dict_nb_test_correct_predictions[model] = nb_test_correct_predictions
            dict_test_dicts[model] = test_dict
            dict_predictions[model] = predictions
            dict_predict_probas[model] = predict_probas
        # Display results for each model in a new window
        test_result_window = TestResultWindow(
            dict_pipelines,
            dict_test_times,
            dict_nb_test_texts,
            dict_nb_test_ai_texts,
            dict_nb_test_human_texts,
            dict_nb_test_ai_predictions,
            dict_nb_test_human_predictions,
            dict_nb_test_skips,
            dict_nb_test_correct_predictions,
            dict_test_dicts,
            dict_predictions,
            dict_predict_probas,
            self.data_language,
        )
        self.childWindows.append(test_result_window)
        test_result_window.show()

    def train_validate_test_models(self, selected_models):
        """
        Train, validate, and test the selected models and show the results in a new window.

        Parameters:
            selected_models (list): A list of selected models.
        """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        dict_pipelines = {}
        dict_train_time = {}
        dict_validation_time = {}
        dict_test_times = {}
        dict_train_validation_test_times = {}
        dict_nb_test_texts = {}
        dict_nb_test_ai_texts = {}
        dict_nb_test_human_texts = {}
        dict_nb_test_ai_predictions = {}
        dict_nb_test_human_predictions = {}
        dict_nb_test_skips = {}
        dict_nb_test_correct_predictions = {}
        dict_validation_dicts = {}
        dict_test_dicts = {}
        dict_predictions = {}
        dict_predict_probas = {}

        for model in selected_models:
            self.pipeline.setClassifierName(model)
            self.pipeline.setCustomPipeline()
            (
                train_time,
                validation_time,
                test_time,
                train_validate_test_time,
                nb_test_texts,
                nb_test_ai_texts,
                nb_test_human_texts,
                nb_test_ai_predictions,
                nb_test_human_predictions,
                nb_test_skips,
                nb_test_correct_predictions,
                validation_dict,
                test_dict,
                predictions,
                predict_probas,
            ) = train_validate_test_single(model, self.pipeline, self.data_language)

            # Store results in dictionaries
            dict_pipelines[model] = self.pipeline
            dict_train_time[model] = train_time
            dict_validation_time[model] = validation_time
            dict_test_times[model] = test_time
            dict_train_validation_test_times[model] = train_validate_test_time
            dict_nb_test_texts[model] = nb_test_texts
            dict_nb_test_ai_texts[model] = nb_test_ai_texts
            dict_nb_test_human_texts[model] = nb_test_human_texts
            dict_nb_test_ai_predictions[model] = nb_test_ai_predictions
            dict_nb_test_human_predictions[model] = nb_test_human_predictions
            dict_nb_test_skips[model] = nb_test_skips
            dict_nb_test_correct_predictions[model] = nb_test_correct_predictions
            dict_validation_dicts[model] = validation_dict
            dict_test_dicts[model] = test_dict
            dict_predictions[model] = predictions
            dict_predict_probas[model] = predict_probas

        # Display results for each models in a new window
        train_validate_test_result_window = TrainValidateTestResultWindow(
            dict_pipelines,
            dict_train_time,
            dict_validation_time,
            dict_test_times,
            dict_train_validation_test_times,
            dict_nb_test_texts,
            dict_nb_test_ai_texts,
            dict_nb_test_human_texts,
            dict_nb_test_ai_predictions,
            dict_nb_test_human_predictions,
            dict_nb_test_skips,
            dict_nb_test_correct_predictions,
            dict_validation_dicts,
            dict_test_dicts,
            dict_predictions,
            dict_predict_probas,
        )
        self.childWindows.append(train_validate_test_result_window)
        train_validate_test_result_window.show()
    
    def get_validation_metrics_from_saved_metrics_file(self, selected_models):
        """Extract validation metrics from saved metrics file and display them in a new window.

        Args:
            selected_models (__List__): The selected models.
        """
        if selected_models == []:
            logger.error("No model selected.")
            QMessageBox.critical(self, "Error", "No model selected.")
            return
        dict_validation_dicts = {}
        for model in selected_models:
            path_to_metrics_file = os.path.join(os.getcwd(), f"code/backend/models/{self.data_language}", model + " Metrics.joblib")
            try:
                metrics_dict = load(path_to_metrics_file)
            except Exception as e:
                logger.error(f"Error while loading metrics file: {e}")
                QMessageBox.critical(self, "Error", f"Error while loading metrics file: {e}")
                return
            dict_validation_dicts[model] = metrics_dict
        validation_result_window = ValidationResultWindow(selected_models=selected_models, dict_validation_time=None, dict_validation_dicts=dict_validation_dicts)
        self.childWindows.append(validation_result_window)
        validation_result_window.show()

    def closeEvent(self, event):
        """
        Close the windows associated with the TrainValidateTestWindow.

        Parameters:
            event: The close event.
        """
        for window in self.childWindows:
            window.close()
        for window in self.childPlots:
            window.close()

    def start_action(self):
        """
        Start the selected action.
        """
        show_load_label = True

        # Get selected action
        action_index = self.action_combo.currentIndex()
        selected_action = self.action_combo.itemText(action_index)

        # Get selected models
        selected_models = []
        for checkbox in self.findChildren(QCheckBox):
            if checkbox.isChecked():
                selected_models.append(checkbox.text())

        try:

            if selected_models == []:
                show_load_label = False
                raise Exception("No model selected.")

            if show_load_label:
                self.start_button.hide()
                self.loading_label.show()
                QApplication.processEvents()

            # Perform action
            if selected_action == "Train (using the configuration file)":
                self.train_models(selected_models)
            elif selected_action == "Learning curve (using the configuration file)":
                self.plot_learning_curve_models(selected_models)
            elif selected_action == "Grid search (using the configuration file)":
                self.grid_search_models(selected_models)
            elif selected_action == "Validate (loading existing model)":
                self.validate_models(selected_models)
            elif selected_action == "Test (loading existing model)":
                self.test_models(selected_models)
            elif (selected_action== "Train, validate and test (using the configuration file)"):
                self.train_validate_test_models(selected_models)
            elif (selected_action == "Get validation metrics from saved metrics file"):
                self.get_validation_metrics_from_saved_metrics_file(selected_models)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
        finally:
            # Remove the loading message
            self.loading_label.hide()
            # Show the start_button again
            self.start_button.show()
