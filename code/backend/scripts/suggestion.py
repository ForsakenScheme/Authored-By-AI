import os
import configparser

from joblib import dump, load
from backend.scripts.pipelines import UserConfigPipeline
from backend.utils.formating import draw_title_box
from utils.log import get_logger
from time import time
from sklearn.pipeline import FeatureUnion, Pipeline

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

def grid_search_best_pipeline(pipeline: UserConfigPipeline) -> tuple[dict, float, str, float]:
    """
    Perform a grid search to find the best pipeline steps to get the best score for a given metric.

    Parameters:
        pipeline (UserConfigPipeline): The pipeline object used for grid search.

    Returns:
        grid_search_time (float): The time taken to perform the grid search in seconds.
        
    """
    start = time()

    # Perform grid_search_best_of_all
    best_parameters, best_score, best_classifier = pipeline.grid_search_best_pipeline()
    grid_search_time = time() - start

    return best_parameters, best_score, best_classifier, grid_search_time

def grid_search_best_model(pipeline: UserConfigPipeline, metric: str) -> tuple[dict, float, str, float]:
    """
    Perform a grid search to find the best model based on the user's config choice.

    Parameters:
        pipeline (UserConfigPipeline): The pipeline object used for grid search.

    Returns:
        grid_search_time (float): The time taken to perform the grid search in seconds.
    """
    start = time()
    # Perform grid_search_best_of_all
    best_parameters, best_score, best_classifier = pipeline.grid_search_best_model(metric)
    grid_search_time = time() - start

    return best_parameters, best_score, best_classifier, grid_search_time

class SuggestionWindow(QMainWindow):
    """
    A QMainWindow for suggestions such as the best model resulting from grid search on all models or on all pipelines.

    Attributes:
        pipeline: The machine learning pipeline.

    Methods:
        setup_pipeline: Set up the machine learning pipeline.
        check_config: Check the pipeline configuration for consistency.
        closeEvent: Close the windows associated with the TrainValidateTestWindow.
        start_action: Start the selected action.
    """
    errorOccurred = pyqtSignal(str)

    def __init__(self, localization_config: configparser.ConfigParser):
        super().__init__()
        self.setWindowTitle("Suggestions")
        self.setGeometry(200, 200, 800, 600)
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
        self.layout.setContentsMargins(50, 50, 50, 50)

        # Title
        title_label = QLabel("Perform grid search to find the best model or pipeline for the given scoring metric.")
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: #333;")
        self.layout.addWidget(title_label)
        self.layout.addSpacing(10)

        # Description
        description_label = QLabel(
            "Grid search will be based of the user configuration file during search of best model. During search of best pipeline, the process is independant of the user configuration file. \nSelect the scoring metric you want to use during grid search by ticking the corresponding box down below:"
        )
        description_label.setStyleSheet("color: #666;")
        self.layout.addWidget(description_label)
        self.layout.addSpacing(10)

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
            "Grid Search for the best MODEL (uses user configuration file) using selected metrics",
            "Grid Search for the best PIPELINE using selected metrics",
            "Extract settings for the best SAVED models for each metric"
        ]
        self.action_combo.addItems(self.actions)
        self.action_combo.setCursor(Qt.PointingHandCursor)
        self.layout.addWidget(self.action_combo)
        self.layout.addSpacing(50)

        # Scoring selection checkboxes
        scoring = ["Accuracy", "Precision", "F1", "Recall"]
        scoring_layout = QHBoxLayout()
        scoring_layout.setSpacing(40)
        scoring_layout.addStretch()

        # Create  columns for the checkboxes
        for metric in scoring:
            checkbox = QCheckBox(metric)
            checkbox.setCursor(Qt.PointingHandCursor)
            scoring_layout.addWidget(checkbox)       
            
        scoring_layout.addStretch()
        self.layout.addLayout(scoring_layout)
        self.layout.addSpacing(80)

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
        
    def start_action(self):
        """
        Start the selected action.
        """
        show_load_label = True

        # Get selected action
        action_index = self.action_combo.currentIndex()
        selected_action = self.action_combo.itemText(action_index)

        # Get selected models
        selected_metrics = []
        for checkbox in self.findChildren(QCheckBox):
            if checkbox.isChecked():
                selected_value = checkbox.text().lower()
                selected_metrics.append(selected_value)

        try:
            if selected_metrics == []:
                show_load_label = False
                QMessageBox().warning(
                    self, "No scoring selected.", "Please select at least one scoring metric."
                )

            if show_load_label:
                self.start_button.hide()
                self.loading_label.show()
                QApplication.processEvents()

            # Perform action
            if selected_action == "Grid Search for the best MODEL (uses user configuration file) using selected metrics":
                self.grid_search_best_models(selected_metrics)
            elif selected_action == "Grid Search for the best PIPELINE using selected metrics":
                self.grid_search_best_pipelines(selected_metrics)
            elif selected_action == "Extract settings for the best SAVED models for each metric":
                self.extract_best_models_values(selected_metrics)
            else:
                QMessageBox.critical(self, "Error", "Invalid action selected.")
                raise Exception("Invalid action selected.")

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
        finally:
            # Remove the loading message
            self.loading_label.hide()
            # Show the start_button again
            self.start_button.show()
            
    def grid_search_best_models(self, selected_metrics):
        """
        Perform a grid search to find the best model based on the user's config choice for pipeline steps.

        Parameters:
            selected_metrics (List[str]): The selected scoring metrics.
        """
        self.check_config()
        self.pipeline.setCustomPipeline()
        best_parameters_dict = {}
        best_score_dict = {}
        best_estimator_dict = {}
        grid_search_time_dict = {}
        
        for metric in selected_metrics:
            # Perform grid search
            logger.info(f"\n{draw_title_box(f"Performing grid search for {metric}", 4)}")
            best_parameters_dict[metric], best_score_dict[metric], best_estimator_dict[metric], grid_search_time_dict[metric] = grid_search_best_model(self.pipeline, metric)
        # Display the results
        self.best_model_result_window = BestModelResultWindow(
            grid_search_time_dict=grid_search_time_dict,
            best_parameters_dict=best_parameters_dict,
            best_score_dict=best_score_dict,
            best_estimator_dict=best_estimator_dict,
            selected_metrics=selected_metrics,
            data_language=self.data_language
        )
        self.best_model_result_window.show()
        
    def grid_search_best_pipelines(self, selected_scorings):
        """
        Perform a grid search to find the best pipeline steps.

        Parameters:
            selected_scorings (List[str]): The selected scoring metrics.
        """
        self.check_config()

        # Perform grid search
        best_parameters, best_score, best_classifier, grid_search_time = grid_search_best_pipeline(self.pipeline)

        # Display the results
        self.best_model_result_window = BestPipelineResultWindow(
            grid_search_time, best_parameters, best_score, best_classifier, selected_scorings, self.data_language
        )
        self.best_model_result_window.show()
        
    def extract_best_models_values(self, selected_metrics):
        """Extracts the best models' parameters for the selected metrics."""
        
        # Initialize a dictionary to store all metrics' information
        pipelines_info = {}
        
        # Mapping of classifier names to user-friendly names
        model_name_mapping = {
            "MultinomialNB": "Multinomial Naive Bayes",
            "SVC": "Support Vector Machine",
            "LogisticRegression": "Logistic Regression",
            "DecisionTreeClassifier": "Decision Tree",
            "RandomForestClassifier": "Random Forest",
            "GradientBoostingClassifier": "Gradient Boosting",
        }

        for metric in selected_metrics:
            model_path = os.path.join(os.getcwd(), f"code/backend/models/{self.data_language}/best/{metric}")
            model_joblib_files = [f for f in os.listdir(model_path) if f.endswith('.joblib') and not f.endswith('_score.joblib')]
            score_joblib_files = [f for f in os.listdir(model_path) if f.endswith('_score.joblib')]
            
            pipeline_path = os.path.join(model_path, model_joblib_files[0])
            try:
                pipeline = load(pipeline_path)
            except Exception as e:
                logger.error(f"Error while loading pipeline: {e}")
                self.errorOccurred.emit(str(e))
                QMessageBox.critical(
                    self,
                    "Error",
                    'Error while loading pipeline.\n\nProbably a bad path setup. You can check the logs in the "code/backend/logs/" folder.',
                )
                return
            try:
                score_path = os.path.join(model_path, score_joblib_files[0])
                score = load(score_path)
            except Exception as e:
                logger.error(f"Error while loading score: {e}")
                self.errorOccurred.emit(str(e))
                QMessageBox.critical(
                    self,
                    "Error",
                    'Error while loading score.\n\nProbably a bad path setup. You can check the logs in the "code/backend/logs/" folder.',
                )
                return

            # Get the classifier from the pipeline's 'classification' step
            classifier = pipeline.named_steps["classification"]
            classifier_name = classifier.__class__.__name__

            # Handle Logistic Regression special cases for L1 and L2 regularization
            if classifier_name == "LogisticRegression":
                penalty = classifier.get_params().get("penalty", "l2")
                if penalty == "l1":
                    estimator_name = "Logistic Regression (L1)"
                else:
                    estimator_name = "Logistic Regression (L2)"
            # Handle ensemble methods like BaggingClassifier
            elif classifier_name == "BaggingClassifier":
                base_classifier_name = classifier.get_params()["estimator"].__class__.__name__
                base_name = model_name_mapping.get(base_classifier_name, base_classifier_name)
                estimator_name = f"Bagging {base_name}"
            # Handle ensemble methods like StackingClassifier
            elif classifier_name == "StackingClassifier":
                final_estimator_name = classifier.get_params()["final_estimator"].__class__.__name__
                final_name = model_name_mapping.get(final_estimator_name, final_estimator_name)
                estimator_name = f"Stacking {final_name}"
            # Use the default mapping for other classifiers
            else:
                estimator_name = model_name_mapping.get(classifier_name, classifier_name)
                
            # Extract classifier parameters
            for step_name, step in pipeline.steps:
                if hasattr(step, 'get_params'):
                    params = step.get_params()
                    if step_name == 'classification':
                        classifier = step
                        classifier_name = classifier.__class__.__name__

                        if classifier_name == "StackingClassifier":
                            classifier_params = self.get_stacking_classifier_params(classifier, model_name_mapping)

                        else:
                            classifier_params = self.format_params(params)

            # Extract transformers parameters
            transformer_params = {}
            for step_name, step in pipeline.steps:
                if hasattr(step, 'transformers'):
                    for transformer_name, transformer in step.transformers:
                        if hasattr(transformer, 'get_params'):
                            transformer_params[transformer_name] = self.format_params(transformer.get_params())

            # Store extracted information in the pipelines_info dictionary
            pipelines_info[metric] = {
                "classifier_name": estimator_name,
                "classifier_params": classifier_params,
                "transformer_params": transformer_params,
                "best_score": score
            }
        # Display the info of each pipeline for each selected metric 
        self.best_pipelines_per_metric_info_window = BestModelsInfoWindow(pipelines_info, self.data_language)
        self.best_pipelines_per_metric_info_window.show()

    def get_stacking_classifier_params(self, classifier, model_name_mapping):
        """Extract parameters for a Stacking Classifier and its estimators."""
        stacking_params = {}
        
        # Get base estimators
        estimators = classifier.get_params()['estimators']
        
        for name, estimator in estimators:
            estimator_name = model_name_mapping.get(estimator.__class__.__name__, estimator.__class__.__name__)
            estimator_params = estimator.get_params()

            # Filter out verbose parameters and format output
            formatted_params = self.format_params(estimator_params)
            
            stacking_params[estimator_name] = formatted_params
            
        # Include other stacking classifier parameters
        stacking_params["cv"] = classifier.get_params()['cv']
        stacking_params["n_jobs"] = classifier.get_params()['n_jobs']
        stacking_params["passthrough"] = classifier.get_params()['passthrough']
        stacking_params["stack_method"] = classifier.get_params()['stack_method']
        
        return stacking_params

    def format_params(self, params):
        """Formats parameters into a vertical structure, excluding 'verbose'."""
        formatted_params = {}
        for param_name, param_value in params.items():
            if param_name != "verbose" and param_name != "base_estimator" and param_name != "estimator__verbose":
                formatted_params[param_name] = param_value
        return formatted_params

class BestModelsInfoWindow(QDialog):
    def __init__(self, pipelines_info, data_language):
        super().__init__()
        self.pipelines_info = pipelines_info
        self.setWindowTitle("Validation Results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setGeometry(0, 0, 800, 800)
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)
        
        self.data_language = data_language
        content = ""

        for metric in self.pipelines_info.keys():
            content += f"<pre><b>Pipeline configuration for best {metric.upper()} model.</b>\n\n"
            
            # Get classifier info to include full name for bagging classifiers
            classifier_info = pipelines_info[metric]['classifier_name']
            content += f"Best classifier for <b>{metric}</b>: {classifier_info}\n\n"
            content += f"Recommended <b>parameter configuration</b> to achieve a <b>score</b> of {pipelines_info["best_score_dict"][metric]} %:\n\n"

            # Display classifier parameters
            classifier_params = pipelines_info[metric]["classifier_params"]
            # Check if it is a bagging or stacking classifier
            if 'estimator' in classifier_params or 'base_estimator' in classifier_params:
                # Distinguish between Bagging and Stacking classifiers
                if 'estimator' in classifier_params:
                    base_estimator_names = classifier_params['estimator'].__class__.__name__
                    content += f"Bagging with <b>base estimator</b>: <b>{base_estimator_names}</b>\n\n"
                else:
                    # Handle more complex stacking setups
                    content += f"<b>Classifier Type:</b> Stacking\n\n"

                content += "<b>Base Estimator Parameters:</b>\n"
                for param, value in classifier_params.items():
                    if 'estimator__' in param:
                        base_param = param.replace('estimator__', '')
                        content += f"\t<b>{base_param}</b>: {value} \n"
                    elif param not in ['estimator', 'base_estimator']:
                        content += f"\t<b>{param}</b>: {value} \n"
                content += "\n"

            else:
                # For non-stacking classifiers
                content += "<b>Classifier Parameters:</b>\n"
                for classifier_name, params in classifier_params.items():
                    content += f"\t<b>{classifier_name}</b>: "

                    # If params is a dictionary, format each key-value pair in the same line
                    if isinstance(params, dict):
                        param_strings = [f"<b>{parameter}</b>: {value}" for parameter, value in params.items()]
                        content += ", ".join(param_strings)
                    else:
                        # Directly append the value if it's not a dictionary
                        content += f"{params}"

                    content += "\n"


            # Display transformer parameters
            transformer_params = pipelines_info[metric]["transformer_params"]
            for transformer_name, params in transformer_params.items():
                content += f"<b>{transformer_name}</b> Configuration:\n\n"
                for parameter, value in params.items():
                    content += f"\t<b>{parameter}</b>: {value} \n"
                content += "\n"

            content += "</pre>\n\n==================================================================================\n\n"

        content_label = QLabel(content)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(content_label)

        content_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        scroll_area.setWidget(content_widget)
        self.layout().addWidget(scroll_area)

class BestModelResultWindow(QDialog):
    """
    A QDialog window to display the results of grid searching for the best model for a given scoring metric.

    Parameters:
        selected_models (List[str]): The selected models.
        dict_grid_search_time (Dict[str, float]): The time taken to perform the grid search for each model.
        dict_grid_search_lists (Dict[str, List[Tuple[Tuple[str, float], Dict[str, Any]]]): The grid search results for each model.
    """
    def __init__(
        self,
        grid_search_time_dict,
        best_parameters_dict,
        best_score_dict,
        best_estimator_dict,
        selected_metrics,
        data_language,
    ):
        super().__init__()
        self.setWindowTitle("Validation results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setGeometry(0, 0, 800, 800)
        content_widget.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)
        
        self.best_parameters_dict = best_parameters_dict
        self.best_score_dict = best_score_dict
        self.best_estimator_dict = best_estimator_dict
        self.grid_search_time_dict = grid_search_time_dict
        self.selected_metrics = selected_metrics
        self.data_language = data_language
        content = ""
        for metric in self.selected_metrics:
            minutes = int(self.grid_search_time_dict[metric] // 60)
            seconds = self.grid_search_time_dict[metric] % 60
            content += f"<pre><b>Grid search results in current user configuration for {metric.upper()}</b>\n\n"
            content += f"Best cross-validation score for <b>{metric}</b>: {best_score_dict[metric] * 100:.2f}%\n"
            content += f"Recommended parameter configuration to achieve this:\n\n"
            for parameter, value in best_parameters_dict[metric].items():
                content += f"<b>\t{parameter}</b>: {value}\n"
            content += "\n"
            content += (
                f"\tGrid search performed in {minutes} minutes and {seconds:.2f} seconds.\n\n"
            )            
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
        
    def closeEvent(self, event):
        """Display a message box for save confirmation when the user tries to exit the application."""
        reply = QMessageBox.question(
            self,
            "Save models",
            "Would you like to save the best models before you leave?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.save_models()
            QMessageBox.information(self, "Success", "Models have been saved successfully.")
        else:
            event.ignore()   
            
    def save_models(self):
        """
        Save the best models for each metric to their respective files using joblib.
        This will delete any existing model files in the metric directory before saving the new model.
        """
        # Define the base path for saving models
        models_path = os.path.join(os.getcwd(), "code/backend/models", self.data_language, "best")

        model_name_mapping = {
            "DecisionTreeClassifier": "Decision Tree",
            "GradientBoostingClassifier": "Gradient Boosting",
            "LogisticRegression": "Logistic Regression",
            "MultinomialNB": "Multinomial Naive Bayes",
            "RandomForestClassifier": "Random Forest",
            "SVC": "Support Vector Machine",
        }

        for metric, estimator in self.best_estimator_dict.items():
            # Get the classifier from the pipeline's 'classification' step
            classifier = estimator.named_steps["classification"]
            classifier_name = classifier.__class__.__name__

            # Handle Logistic Regression special cases for L1 and L2 regularization
            if classifier_name == "LogisticRegression":
                penalty = classifier.get_params().get("penalty", "l2")
                if penalty == "l1":
                    estimator_name = "Logistic Regression (L1)"
                else:
                    estimator_name = "Logistic Regression (L2)"
            # Handle ensemble methods like BaggingClassifier
            elif classifier_name == "BaggingClassifier":
                base_classifier_name = classifier.get_params()["estimator"].__class__.__name__
                base_name = model_name_mapping.get(base_classifier_name, base_classifier_name)
                estimator_name = f"Bagging {base_name}"
            # Handle ensemble methods like StackingClassifier
            elif classifier_name == "StackingClassifier":
                final_estimator_name = classifier.get_params()["final_estimator"].__class__.__name__
                final_name = model_name_mapping.get(final_estimator_name, final_estimator_name)
                estimator_name = f"Stacking {final_name}"
            # Use the default mapping for other classifiers
            else:
                estimator_name = model_name_mapping.get(classifier_name, classifier_name)

            # Define the directory path for the metric
            metric_directory = os.path.join(models_path, metric.lower())
            os.makedirs(metric_directory, exist_ok=True)

            # Remove all files in the metric directory
            for file in os.listdir(metric_directory):
                file_path = os.path.join(metric_directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted existing file: {file} in {metric_directory}.")

            # Define the full path for saving the model
            model_filename = os.path.join(metric_directory, f"{estimator_name}.joblib")
            best_score = self.best_score_dict[metric]
            
            # Save the fitted model to a file
            try:
                dump(estimator, model_filename)
                logger.info(f"Successfully saved {estimator_name} for {metric} to {model_filename}.")
                                
            except Exception as e:
                logger.error(f"Error while saving model: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    'Error while saving model.\n\nProbably a bad path setup. You can check the logs in the "code/backend/logs/" folder.',
                )
                return
            # Save the best score to a file with the same name as the model
            try: 
                dump(best_score, os.path.join(metric_directory, f"{estimator_name}_score.joblib"))
                logger.info(f"Successfully saved {estimator_name} score for {metric} to {model_filename}.")
                
            except Exception as e:
                logger.error(f"Error while saving score: {e}")
                QMessageBox.critical(
                    self,
                    "Error",
                    'Error while saving score.\n\nProbably a bad path setup. You can check the logs in the "code/backend/logs/" folder.',
                )
                return          
                    
class BestPipelineResultWindow(QDialog):
    """
    A QDialog window to display the results of grid searching for the best model for a given scoring metric.

    Parameters:
        selected_models (List[str]): The selected models.
        dict_grid_search_time (Dict[str, float]): The time taken to perform the grid search for each model.
        dict_grid_search_lists (Dict[str, List[Tuple[Tuple[str, float], Dict[str, Any]]]): The grid search results for each model.
    """
    def __init__(
        self,
        grid_search_time_dict,
        best_parameters_dict,
        best_score_dict,
        best_estimator_dict,
        selected_metrics,
    ):
        super().__init__()
        self.setWindowTitle("Validation results")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Set margins for the main window
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        self.setLayout(main_layout)

        # Create content widget
        content_widget = QWidget()
        content_widget.setGeometry(0, 0, 800, 800)
        content_widget.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        content_widget.setContentsMargins(10, 10, 10, 10)
        content_layout = QVBoxLayout(content_widget)
        
        self.best_parameters_dict = best_parameters_dict
        self.best_score_dict = best_score_dict
        self.best_estimator_dict = best_estimator_dict
        self.grid_search_time_dict = grid_search_time_dict
        self.selected_metrics = selected_metrics
        content = ""
        for metric in self.selected_metrics:
            minutes = int(self.grid_search_time_dict[metric] // 60)
            seconds = self.grid_search_time_dict[metric] % 60
            content += f"<pre><b>Grid search results in current user configuration for {self.best_estimator_dict[metric]}</b>\n\n"
            content += f"Best cross-validation score for <b>{metric}</b>: {best_score_dict[metric] * 100:.2f}%\n"
            content += f"Recommended parameter configuration to achieve this:\n\n"
            for parameter, value in best_parameters_dict[metric].items():
                content += f"\t{parameter}: {value}\n"
            content += "\n"
            content += (
                f"\tGrid search performed in {minutes} minutes and {seconds:.2f} seconds.\n\n"
            )            
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