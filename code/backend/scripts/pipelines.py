import configparser

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from backend.utils.formating import draw_title_box
from backend.utils.log import get_logger
from backend.utils.configurating import config
from backend.scripts.database_functions import (
    get_all_texts_and_labels_from_table,
    get_nb_of_processed_texts,
)
from backend.scripts.preprocessing import (
    split_data,
    FeatureExtractorBeforePreprocessing,
    Preprocessing,
    FeatureExtractorAfterPreprocessing,
    FeatureSelection,
)

from joblib import load, dump
from pathlib import Path

logger = get_logger(__name__)

class UserConfigPipeline:
    """
    Class representing a user-configured pipeline for text classification.

    Attributes:
        random_state (int): Random state for reproducibility.
        config (ConfigParser): Configuration parser object.
        nb_features (int or str): Number of features to select or "all".
        nb_folds (int): Number of folds for cross-validation.
        classifier_name (str): Name of the selected classifier.
        X_train (array-like): Training data features.
        X_val (array-like): Validation data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_val (array-like): Validation data labels.
        y_test (array-like): Test data labels.
        feature_selector (SelectKBest): Feature selector object.
        custom_pipeline (Pipeline): Custom pipeline with feature selection.
        custom_pipeline_without_feature_selection (Pipeline): Custom pipeline without feature selection.
        param_grid (dict): Hyperparameter grid for grid search.
        logger (Logger): Logger object for logging.

    Methods:
        create_custom_pipeline(feature_selector=True): Creates a custom pipeline based on the user's configuration.
        train(): Trains the custom model then dumps the model to a file.
        validate(): Validates the custom model and calculates metrics on the validation set.
        grid_search(): Performs grid search to find the best hyperparameters for the custom model.
        test(): Tests the custom model on the test set.
        predict_unknown_texts(text): Predicts the label for unknown texts.
    """

    def __init__(self):
        self.localization_config = configparser.ConfigParser(comment_prefixes="#", inline_comment_prefixes="#")
        self.load_localization_config()
        
        self.data_language = self.localization_config.get("Data", "language")
        self.X, self.y = self.get_all_texts_and_labels()
        self.validate_data()
        
        self.config = self.load_user_config()
        self.random_state = self.config.getint("RandomState", "random_state")
        self.nb_folds = self.config.getint("CrossValidation", "nb_folds")
        
        self.validate_configuration()
        
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(self.X, self.y)
        self.custom_pipeline = None
        self.param_grids = create_all_hyperparam_grids()
        self.param_grid = None
        self.classifier_name = "Stacking Support Vector Machine"
        self.metrics = create_scoring()
        logger.info("UserConfigPipeline initialized.")

    def load_localization_config(self):
        """Loads localization configuration."""
        try:
            self.localization_config.read("code/backend/config/localization.ini")
        except Exception as e:
            logger.error("Failed to load localization configuration: %s", e)
            raise

    def get_all_texts_and_labels(self):
        """Fetches texts and labels from the processed table."""
        X, y = get_all_texts_and_labels_from_table("processed", self.data_language)
        if len(X) != len(y) or not X or not y:
            raise ValueError("Texts and labels must have the same non-zero length.")
        return X, y

    def validate_data(self):
        """Validates the fetched data."""
        if len(self.X) != get_nb_of_processed_texts(self.data_language):
            raise ValueError("Number of texts does not match the expected count in the database.")
        if set(self.y) != {"ai", "human"}:
            raise ValueError("Labels must include both 'ai' and 'human'.")

    def load_user_config(self):
        """Loads user configuration from the config file."""
        try:
            return config  # Assuming `config` is defined elsewhere in your code.
        except Exception as e:
            logger.error("Failed to load user configuration: %s", e)
            raise

    def validate_configuration(self):
        """Validates the user configuration."""
        if self.random_state < 0:
            raise ValueError("Random state must be a non-negative integer.")
        if self.nb_folds < 2:
            raise ValueError("Number of folds must be at least 2.")
        nb_features = self.config.get("FeatureSelection", "nb_features")
        if not nb_features.isdigit() and nb_features != "all":
            raise ValueError("Number of features must be a positive integer or 'all'.")
        if nb_features.isdigit() and int(nb_features) <= 0:
            raise ValueError("Number of features must be a positive integer.")

        if self.config.getboolean("Preprocessing", "stemm") and self.config.getboolean("Preprocessing", "lemmatize"):
            raise ValueError("Stemming and lemmatization cannot be both enabled.")

        if self.config.getboolean("Preprocessing", "lemmatize") and self.config.getboolean("Preprocessing", "stemm"):
            raise ValueError("Lemmatization must be disabled if stemming is enabled.")

    def setClassifierName(self, classifier_name):
        """
        Set the classifier name.

        Parameters:
            classifier_name (str): The name of the classifier to set.
        """
        self.classifier_name = classifier_name

    def setCustomPipeline(self):
        """
        Set the custom pipeline based on the user's configuration.
        """
        self.custom_pipeline = self.create_custom_pipeline()

    def setParamGrid(self, classifier_name:str):
        """
        Set the hyperparameter grid for the specified classifier.

        Parameters:
            classifier_name (str): The name of the classifier to set the hyperparameter grid for.
        """
        self.param_grid = self.param_grids[classifier_name]

    def loadCustomPipeline(self, model):
        """
        Load a custom pipeline from a file.

        Parameters:
            model (str): The name of the model file containing the custom pipeline.
        """
        file_path = Path(__file__).resolve().parent.parent.parent / f"backend/models/{self.data_language}/{model}.joblib"
        try:
            model_pipeline = load(file_path)
            logger.info("Successfully loaded custom pipeline from file.")
        except Exception as e:
            logger.error("Failed to load custom pipeline from file: %s", e)
        self.custom_pipeline = model_pipeline

    def create_custom_pipeline(self):
        """
        Creates a custom pipeline based on the user's configuration.

        Returns:
            Pipeline: The custom pipeline.
        """
        width = 40
        title = "Custom Pipeline Creation"
        logger.info(
            "\n{0}\n= {1} =\n{0}".format(
                "=" * width,
                title.center(width - 4)
            )
        )
        logger.info("Creating custom pipeline...")
        if self.classifier_name not in get_all_possible_classifiers():
            raise ValueError(
                f"Classifier {self.classifier_name} not found in the list of available classifiers."
            )
        # get pipeline config based on the .ini file
        pipeline_config = {
            # before pipeline
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
            # after preprocessing
            "textWordCounter": self.config.getboolean(
                "FeatureExtractionAfterPreprocessing", "textWordCounter"
            ),
            "wordLength": self.config.getboolean(
                "FeatureExtractionAfterPreprocessing", "wordLength"
            ),
            "vocabularySize": self.config.getboolean(
                "FeatureExtractionAfterPreprocessing", "vocabularySize"
            ),
            "data_language": self.data_language,
        }
        logger.info(f"pipeline config: {pipeline_config}")
        # create a pipeline with the specified config
        pipeline = create_pipeline(**pipeline_config)
        logger.info("Pipeline created.")
        # determine the classifier based on the user's selection in the ini file
        if self.classifier_name == "Support Vector Machine":
            classifier = SVM(
                C=10,
                degree=2,
                gamma=0.001,
                probability=True,
                kernel="linear",
                random_state=self.random_state,
            )
        elif self.classifier_name == "Logistic Regression (L1)":
            classifier = LogisticRegression(
                C=10,
                max_iter=100,
                solver="liblinear",
                penalty="l1",
                multi_class="auto",
                random_state=self.random_state,
            )
        elif self.classifier_name == "Logistic Regression (L2)":
            classifier = LogisticRegression(
                C=10,
                max_iter=100,
                solver="liblinear",
                penalty="l2",
                multi_class="auto",
                random_state=self.random_state,
            )
        elif self.classifier_name == "Decision Tree":
            classifier = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=30,
                random_state=self.random_state,
                min_samples_leaf=4,
                min_samples_split=2,
            )
        elif self.classifier_name == "Random Forest":
            classifier = RandomForestClassifier(
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=5,
                n_estimators=200,
                criterion="gini",
                random_state=self.random_state,
            )
        elif self.classifier_name == "Gradient Boosting":
            classifier = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                random_state=self.random_state,
            )
        elif self.classifier_name == "Multinomial Naive Bayes":
            classifier = MultinomialNB(alpha=1.0, fit_prior=True)
        elif self.classifier_name == "Stacking Decision Tree":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator=DecisionTreeClassifier(max_depth=5),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Stacking Random Forest":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator=RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Stacking Gradient Boosting":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator=GradientBoostingClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Stacking Multinomial Naive Bayes":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator=MultinomialNB(),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Stacking Support Vector Machine":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator=SVM(probability=True),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Stacking Logistic Regression (L1)":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator = LogisticRegression(
                    C=10,
                    max_iter=100,
                    solver="liblinear",
                    penalty="l1",
                    multi_class="auto",
                    random_state=self.random_state,
                ),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Stacking Logistic Regression (L2)":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator = LogisticRegression(
                    C=10,
                    max_iter=100,
                    solver="liblinear",
                    penalty="l2",
                    multi_class="auto",
                    random_state=self.random_state,
                ),
                stack_method="auto",
                cv=StratifiedKFold(
                    n_splits=self.nb_folds, shuffle=True, random_state=self.random_state
                ),
            )
        elif self.classifier_name == "Bagging Multinomial Naive Bayes":
            classifier = BaggingClassifier(
                MultinomialNB(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            )
        elif self.classifier_name == "Bagging Support Vector Machine":
            classifier = BaggingClassifier(
                SVM(probability=True),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            )
        elif self.classifier_name == "Bagging Logistic Regression":
            classifier = BaggingClassifier(
                LogisticRegression(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            )
        elif self.classifier_name == "Bagging Decision Tree":
            classifier = BaggingClassifier(
                DecisionTreeClassifier(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            )
        elif self.classifier_name == "Bagging Random Forest":
            classifier = BaggingClassifier(
                RandomForestClassifier(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            )
        elif self.classifier_name == "Bagging Gradient Boosting":
            classifier = BaggingClassifier(
                GradientBoostingClassifier(),
                n_estimators=10,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                f"Classifier {self.classifier_name} not found in the list of available classifiers."
            )
        logger.info(f"Selected classifier: {self.classifier_name}")
        # create custom pipeline with feature extraction from Before and After preprocessing, also apply classifier and feature selection
        custom_pipeline = Pipeline(
            [("pipeline", pipeline), ("classification", classifier)]
        )
        logger.info("Added classifier to the custom pipeline. Custom pipeline full steps: \n%s\n", custom_pipeline.named_steps)
        return custom_pipeline

    def train(self, test=False):
        """
        Trains the custom model then dumps the model to a file.

        Returns:
            validation_dict (dict): A dictionary containing the validation metrics.

            best_parameters_for_score (list): A list of tuples containing the best parameters and scores for each scoring metric.
        """
        title_training = f"=    Training model {self.classifier_name}   ="
        title_training_length = len(title_training)
        equal_signs_training = "=" * title_training_length

        # train the custom model
        logger.info(
            f"\n{equal_signs_training}\n{title_training}\n{equal_signs_training}"
        )
        logger.info("Fitting model.")
        self.custom_pipeline = self.custom_pipeline.fit(self.X_train, self.y_train)
        logger.info("Successfully fitted.")

        if test:
            return self.custom_pipeline
        # save the trained model to a file
        try:
            with open(
                ("code/backend/models/" + "/" + self.data_language + "/" + self.classifier_name + ".joblib"), "wb"
            ) as joblib_file:
                dump(self.custom_pipeline, joblib_file)
            logger.info("Successfully dumped trained custom pipeline to file.")
        except Exception as e:
            logger.error("Failed to dump trained custom pipeline to file: %s", e)
            

    def validate(self):
        """
        Validates the custom model and calculates metrics on the validation set.

        Returns:
            validation_dict (dict): A dictionary containing the validation metrics.

            best_parameters_for_score (list): A list of tuples containing the best parameters and scores for each scoring metric.
        """
        title_validation = f"=    Validating model {self.classifier_name}   ="
        title_validation_length = len(title_validation)
        equal_signs_validation = "=" * title_validation_length
        y_pred = self.custom_pipeline.predict(self.X_val)
        # calculate metrics for the validation set
        logger.info(
            f"\n{equal_signs_validation}\n{title_validation}\n{equal_signs_validation}"
        )
        (
            validation_accuracy,
            validation_precision,
            validation_recall,
            validation_f1,
            validation_cm,
        ) = calculate_metrics(self.y_val, y_pred)
        validation_dict = {
            "accuracy": validation_accuracy,
            "precision": validation_precision,
            "recall": validation_recall,
            "f1": validation_f1,
            "cm": validation_cm,
        }
        logger.info("Successfully validated custom model...")
        logger.info(f"Validation accuracy: {validation_accuracy}")
        logger.info(f"Validation precision: {validation_precision}")
        logger.info(f"Validation recall: {validation_recall}")
        logger.info(f"Validation F1 score: {validation_f1}")
        logger.info(f"Validation confusion matrix:\n{validation_cm}")
        return validation_dict

    def grid_search(self):
        """
        Performs grid search to find the best hyperparameters for the custom model using a given metric. Also calculate the best score for each scoring metric using the best parameters for selected scoring.

        Returns:
            best_parameters_for_metric (list): A list of tuples containing the best parameters and scores for each scoring metric.
        """
        best_parameters_for_score = []
        for metric in self.metrics:
            logger.info(f"{draw_title_box(f"Performing grid search for {metric}")}")
            best_params, best_score = perform_grid_search(
                self.custom_pipeline,
                metric[1],
                self.param_grid,
                self.X_train,
                self.y_train,
                self.random_state,
                self.nb_folds,
            )
            best_parameters_for_score.append(
                ((metric[0], best_score), best_params)
            )  # example (("precision", 0.88)), {"classification__C": 1, "classification__kernel": "linear"})
        return best_parameters_for_score
    
    def grid_search_best_model(self, scoring_metric="accuracy"):
        """Performs grid search to find the best model and it's best hyper-parameters given a scoring metric used for evaluation of the custom model.

        Args:
            scoring_metric (string): name of the scoring metric to be used for evaluation of the custom model (accuracy, precision, recall, f1).
        
        Returns:
            dict: A dictionary containing the result of the grid search.
            The dictionary has the following keys:\n
                - "best_parameters" (dict): The best hyperparameters to give the custom model.
                - "best_score" (float): The best score obtained by the custom model.
                - "best_classifier" (str): The name of the best classifier.
        """
        if scoring_metric == "accuracy":
            metric = self.metrics[0][1]
        elif scoring_metric == "precision":
            metric = self.metrics[1][1]
        elif scoring_metric == "recall":
            metric = self.metrics[2][1]
        elif scoring_metric == "f1":
            metric = self.metrics[3][1]
        else:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}")
        
        best_parameters, best_score, best_classifier = perform_grid_search(
            self.custom_pipeline,
            metric,
            create_all_hyperparam_grids(True),
            self.X_train,
            self.y_train,
            self.random_state,
            self.nb_folds,
            return_best_classifier=True,            
        )
        return best_parameters, best_score, best_classifier
    
    def grid_search_best_pipeline(self, scoring_metric="accuracy"):
        """Performs grid search to find the best pipeline steps and it's best hyper-parameters given a scoring metric used for evaluation of the custom model.

        Args:
            scoring_metric (string): name of the scoring metric to be used for evaluation of the custom model (accuracy, precision, recall, f1).
        
        Returns:
            dict: A dictionary containing the result of the grid search.
            The dictionary has the following keys:\n
                - "best_parameters" (dict): The best hyperparameters to give the custom model.
                - "best_score" (float): The best score obtained by the custom model.
                - "best_pipeline" (Pipeline): The best pipeline.
        """
        if scoring_metric == "accuracy":
            metric = self.metrics[0][1]
        elif scoring_metric == "precision":
            metric = self.metrics[1][1]
        elif scoring_metric == "recall":
            metric = self.metrics[2][1]
        elif scoring_metric == "f1":
            metric = self.metrics[3][1]
        else:
            raise ValueError(f"Invalid scoring metric: {scoring_metric}")
        
        best_parameters, best_score, best_pipeline = perform_grid_search_pipeline(
            self.custom_pipeline,
            metric,
            self.param_grid,
            self.X_train,
            self.y_train,
            self.random_state,
            self.nb_folds,
            return_best_pipeline=True,            
        )
        return best_parameters, best_score, best_pipeline
        
        
    def test(self):
        """
        Calculate metrics for the test set.

        Returns:
            dict: A dictionary containing the calculated metrics for the test set.
            The dictionary has the following keys:\n
                - "accuracy": The test accuracy.
                - "precision": The test precision.
                - "recall": The test recall.
                - "f1": The test F1 score.
        """
        title = f"=    Testing model {self.classifier_name}   ="
        title_length = len(title)
        equal_signs = "=" * title_length
        logger.info(f"\n{equal_signs}\n{title}\n{equal_signs}")
        nb_test_ai_predictions = 0
        nb_test_human_predictions = 0
        nb_test_ai_texts = 0
        nb_test_human_texts = 0
        nb_test_skips = 0
        nb_test_correct_predictions = 0
        test_predictions = self.custom_pipeline.predict(self.X_test)
        test_accuracy, test_precision, test_recall, test_f1, cm = calculate_metrics(
            self.y_test, test_predictions
        )
        test_dict = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "cm": cm,
        }
        logger.info("Successfully tested custom model...")
        logger.info(f"Test accuracy: {test_accuracy}")
        logger.info(f"Test precision: {test_precision}")
        logger.info(f"Test recall: {test_recall}")
        logger.info(f"Test F1 score: {test_f1}")
        logger.info(f"Test confusion matrix:\n{cm}")
        logger.info("Extracting test prediction probabilities...")
        test_predict_probas = self.custom_pipeline.predict_proba(self.X_test)
        for label, prediction in zip(self.y_test, test_predictions):
            if label == "ai":
                nb_test_ai_texts += 1
                if prediction == "ai":
                    nb_test_correct_predictions += 1
                    nb_test_ai_predictions += 1
                if prediction == "human":
                    nb_test_human_predictions += 1
            elif label == "human":
                nb_test_human_texts += 1
                if prediction == "human":
                    nb_test_correct_predictions += 1
                    nb_test_human_predictions += 1
                if prediction == "ai":
                    nb_test_ai_predictions += 1
            else:
                if prediction == "skipped (too short)":
                    nb_test_skips += 1
                else:
                    raise ValueError(f"Invalid prediction: {prediction}")
        return (
            nb_test_ai_texts,
            nb_test_human_texts,
            nb_test_ai_predictions,
            nb_test_human_predictions,
            nb_test_skips,
            nb_test_correct_predictions,
            test_dict,
            test_predictions,
            test_predict_probas,
        )


def create_base_classifiers(random_state=42):
    """
    Creates a dictionary of base classifiers for stacking and bagging.

    Returns:
        dict: A dictionary containing the base classifiers.
    """
    base_classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0, fit_prior=True),
        "Support Vector Machine": SVM(
            C=10,
            degree=2,
            gamma=0.001,
            probability=True,
            kernel="linear",
            random_state=random_state,
        ),
        "Logistic Regression (L1)": LogisticRegression(
            C=10,
            max_iter=100,
            solver="liblinear",
            penalty="l2",
            multi_class="auto",
            random_state=random_state,
        ),
        "Logistic Regression (L2)": LogisticRegression(
            C=10,
            max_iter=100,
            solver="liblinear",
            penalty="l2",
            multi_class="auto",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=30,
            random_state=random_state,
            min_samples_leaf=4,
            min_samples_split=2,
        ),
        "Random Forest": RandomForestClassifier(
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=200,
            criterion="gini",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            random_state=random_state,
        ),
    }
    return base_classifiers


def create_final_estimators(random_state=42):
    """
    Creates and returns a dictionary of final estimators for stacking classifiers.

    Returns:
        dict: A dictionary containing the final estimators.
    """
    final_estimators = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Logistic Regression (L1)": LogisticRegression(
            C=10,
            max_iter=100,
            solver="liblinear",
            penalty="l1",
            multi_class="auto",
            random_state=random_state,
        ),
        "Logistic Regression (L2)": LogisticRegression(
            C=10,
            max_iter=100,
            solver="liblinear",
            penalty="l2",
            multi_class="auto",
            random_state=random_state,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVM(probability=True, random_state=random_state),
    }
    return final_estimators


def create_stacking_classifiers(
    base_classifiers: dict, final_estimators: dict, random_state=42
):
    """
    Create stacking classifiers using base classifiers and final estimators.

    Parameters:
        base_classifiers (dict): A dictionary of base classifiers.
        final_estimators (dict): A dictionary of final estimators.

    Returns:
        dict: A dictionary of stacking classifiers.
    """
    stacking_classifiers = {}
    base_classifiers_list = list(base_classifiers.items())
    for estimator_name, final_estimator in final_estimators.items():
        stacking_classifier = StackingClassifier(
            estimators=base_classifiers_list,
            final_estimator=final_estimator,
            stack_method="auto",
            cv=StratifiedKFold(
                n_splits=config.getint("CrossValidation", "nb_folds"),
                shuffle=True,
                random_state=random_state,
            ),
        )
        stacking_classifiers[f"Stacking {estimator_name}"] = stacking_classifier
    return stacking_classifiers


def create_bagging_classifiers(base_classifiers: dict, random_state=42):
    """
    Create bagging classifiers based on the provided base classifiers.

    Parameters:
        base_classifiers (dict): A dictionary containing the base classifiers.

    Returns:
        dict: A dictionary containing the bagging classifiers.
    """
    bagging_classifiers = {}
    for base_name, base_classifier in base_classifiers.items():
        bagging_classifier = BaggingClassifier(
            estimator=base_classifier,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            random_state=random_state,
        )
        bagging_classifiers[f"Bagging {base_name}"] = bagging_classifier
    return bagging_classifiers


def get_all_possible_classifiers():
    """
    Returns a dictionary containing all possible classifiers.

    This function creates base classifiers, final estimator, stacking classifiers, and bagging classifiers using their relative functions.
    It then combines them into a single dictionary and returns it.

    Returns:
        dict: A dictionary containing all possible classifiers.
    """
    base_classifiers = create_base_classifiers()
    final_estimator = create_final_estimators()
    stacking_classifiers = create_stacking_classifiers(
        base_classifiers, final_estimator
    )
    bagging_classifiers = create_bagging_classifiers(base_classifiers)
    classifiers = {**base_classifiers, **stacking_classifiers, **bagging_classifiers}
    return classifiers


def create_pipeline(
    configParser=config,
    stopWords=False,
    errorDetector=False,
    punctuationFrequency=False,
    sentenceLength=False,
    namedEntity=False,
    sentimentAnalysis=False,
    textWordCounter=False,
    wordLength=False,
    vocabularySize=False,
    data_language="english",
):
    """
    Create a pipeline for text data.

    Parameters:
        configParser: Configuration parser object.
        stopWords: Whether to include stop words feature.
        errorDetector: Whether to include error detector feature.
        punctuationFrequency: Whether to include punctuation frequency feature.
        sentenceLength: Whether to include sentence length feature.
        namedEntity: Whether to include named entity feature.
        sentimentAnalysis: Whether to include sentiment analysis feature.
        textWordCounter: Whether to include text word counter feature.
        wordLength: Whether to include word length feature.
        vocabularySize: Whether to include vocabulary size feature.
        data_language: The language of the data.

    Returns:
        sklearn.pipeline.Pipeline: The model pipeline.
    """
    logger.info("Creating pipeline...")
    logger.info("Selected pipeline language: %s", data_language)
    # retrive nb_features from config.ini file
    nb_features = config.get("FeatureSelection", "nb_features")
    if nb_features != "all" and (not nb_features.isdigit()):
        raise ValueError(f'Number of features must be an integer or "all".')
    if nb_features == "all":
        nb_features = nb_features
    else:
        nb_features = int(nb_features)
    feature_extraction_before_preprocessing = FeatureExtractorBeforePreprocessing(
        stopWords,
        errorDetector,
        punctuationFrequency,
        sentenceLength,
        namedEntity,
        sentimentAnalysis,
        data_language,
    )
    feature_extraction_after_preprocessing = FeatureExtractorAfterPreprocessing(
        config=configParser,
        textWordCounter=textWordCounter,
        wordLength=wordLength,
        vocabularySize=vocabularySize,
    )
    preprocessing = Preprocessing(
        configParser.getboolean("Preprocessing", "punctuation"),
        configParser.getboolean("Preprocessing", "stemm"),
        configParser.getboolean("Preprocessing", "lemmatize"),
        data_language,
    )
    steps = [
        (
            "featureExtractionUnion",
            FeatureUnion(
                [
                    (
                        "featureExtractionBeforePreprocessing",
                        feature_extraction_before_preprocessing,
                    ),
                    (
                        "preprocessingAndFeatureExtractionAfterPreprocessing",
                        Pipeline(
                            [
                                ("preprocessing", preprocessing),
                                (
                                    "featureExtractionAfterPreprocessing",
                                    feature_extraction_after_preprocessing,
                                ),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        ("featureSelection", FeatureSelection(k=nb_features)),
    ]
    logger.info(f"Full pipeline config steps without classifier:\n{steps}\n")
    return Pipeline(steps)


def create_all_hyperparam_grids(classifiers=False):
    """
    Create hyperparameter grids for different classifiers.

    Returns:
        dict: A dictionary containing hyperparameter grids for different classifiers.
    """
    if not classifiers:
        hyperparam_grids = {
            "Multinomial Naive Bayes": {
                "classification__alpha": [0.1, 1, 10],
                "classification__fit_prior": [True, False],
            },
            "Support Vector Machine": {
                "classification__C": [0.1, 1, 10],
                "classification__kernel": ["linear", "rbf"],
                "classification__gamma": [0.001, 0.01, 0.1],
                "classification__degree": [2, 3, 4],
                "classification__probability": [True],
            },
            "Logistic Regression (L1)": {
                "classification__penalty": ["l1"],
                "classification__C": [0.1, 1, 10],
                "classification__solver": ["liblinear"],
                "classification__max_iter": [100, 200, 300],
                "classification__multi_class": ["auto", "ovr"],
            },
            "Logistic Regression (L2)": {
                "classification__penalty": ["l2"],
                "classification__C": [0.1, 1, 10],
                "classification__solver": ["liblinear", "lbfgs"],
                "classification__max_iter": [100, 200, 300],
                "classification__multi_class": ["auto", "ovr"],
            },
            "Decision Tree": {
                "classification__criterion": ["gini", "entropy"],
                "classification__splitter": ["best", "random"],
                "classification__max_depth": [None, 10, 20, 30],
                "classification__min_samples_split": [2, 5, 10],
                "classification__min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "classification__n_estimators": [10, 50, 100, 200],
                "classification__criterion": ["gini", "entropy"],
                "classification__max_depth": [None, 10, 20, 30],
                "classification__min_samples_split": [2, 5, 10],
                "classification__min_samples_leaf": [1, 2, 4],
            },
            "Gradient Boosting": {
                "classification__n_estimators": [100, 200, 300],
                "classification__learning_rate": [0.1, 0.01, 0.001],
                "classification__max_depth": [3, 4, 5],
                "classification__subsample": [0.8, 0.9, 1.0],
            },
            "Stacking Multinomial Naive Bayes": {
                "classification__final_estimator__alpha": [0.1, 1, 10],
                "classification__final_estimator__fit_prior": [True, False],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            "Stacking Support Vector Machine": {
                "classification__final_estimator__C": [0.1, 1, 10],
                "classification__final_estimator__kernel": ["linear", "rbf"],
                "classification__final_estimator__gamma": [0.001, 0.01, 0.1],
                "classification__final_estimator__degree": [2, 3, 4],
                "classification__final_estimator__probability": [True],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            "Stacking Logistic Regression (L1)": {
                "classification__final_estimator__penalty": ["l1"],
                "classification__final_estimator__C": [0.1, 1, 10],
                "classification__final_estimator__solver": ["liblinear"],
                "classification__final_estimator__max_iter": [100, 200, 300],
                "classification__final_estimator__multi_class": ["auto", "ovr"],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            "Stacking Logistic Regression (L2)": {
                "classification__final_estimator__penalty": ["l2"],
                "classification__final_estimator__C": [0.1, 1, 10],
                "classification__final_estimator__solver": ["liblinear"],
                "classification__final_estimator__max_iter": [100, 200, 300],
                "classification__final_estimator__multi_class": ["auto", "ovr"],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            "Stacking Decision Tree": {
                "classification__final_estimator__max_depth": [3, 4, 5],
                "classification__final_estimator__min_samples_split": [2, 5, 10],
                "classification__final_estimator__min_samples_leaf": [1, 2, 4],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            "Stacking Random Forest": {
                "classification__final_estimator__n_estimators": [100, 200, 300],
                "classification__final_estimator__criterion": ["gini", "entropy"],
                "classification__final_estimator__max_depth": [3, 4, 5],
                "classification__final_estimator__min_samples_split": [2, 5, 10],
                "classification__final_estimator__min_samples_leaf": [1, 2, 4],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            "Stacking Gradient Boosting": {
                "classification__final_estimator__n_estimators": [100, 200, 300],
                "classification__final_estimator__learning_rate": [0.1, 0.01, 0.001],
                "classification__final_estimator__max_depth": [3, 4, 5],
                "classification__final_estimator__subsample": [0.8, 0.9, 1.0],
            },
            "Bagging Multinomial Naive Bayes": {
                "classification__n_estimators": [10, 50, 100],
                "classification__max_samples": [0.5, 0.7, 1.0],
                "classification__max_features": [0.5, 0.7, 1.0],
                "classification__bootstrap": [True, False],
                "classification__bootstrap_features": [True, False],
                "classification__oob_score": [True, False],
                "classification__warm_start": [True, False],
            },
            "Bagging Support Vector Machine": {
                "classification__n_estimators": [10, 50, 100],
                "classification__max_samples": [0.5, 0.7, 1.0],
                "classification__max_features": [0.5, 0.7, 1.0],
                "classification__bootstrap": [True, False],
                "classification__bootstrap_features": [True, False],
                "classification__oob_score": [True, False],
                "classification__warm_start": [True, False],
            },
            "Bagging Logistic Regression": {
                "classification__n_estimators": [10, 50, 100],
                "classification__max_samples": [0.5, 0.7, 1.0],
                "classification__max_features": [0.5, 0.7, 1.0],
                "classification__bootstrap": [True, False],
                "classification__bootstrap_features": [True, False],
                "classification__oob_score": [True, False],
                "classification__warm_start": [True, False],
            },
            "Bagging Decision Tree": {
                "classification__n_estimators": [10, 50, 100],
                "classification__max_samples": [0.5, 0.7, 1.0],
                "classification__max_features": [0.5, 0.7, 1.0],
                "classification__bootstrap": [True, False],
                "classification__bootstrap_features": [True, False],
                "classification__oob_score": [True, False],
                "classification__warm_start": [True, False],
            },
            "Bagging Random Forest": {
                "classification__n_estimators": [10, 50, 100],
                "classification__max_samples": [0.5, 0.7, 1.0],
                "classification__max_features": [0.5, 0.7, 1.0],
                "classification__bootstrap": [True, False],
                "classification__bootstrap_features": [True, False],
                "classification__oob_score": [True, False],
                "classification__warm_start": [True, False],
            },
            "Bagging Gradient Boosting": {
                "classification__n_estimators": [10, 50, 100],
                "classification__max_samples": [0.5, 0.7, 1.0],
                "classification__max_features": [0.5, 0.7, 1.0],
                "classification__bootstrap": [True, False],
                "classification__bootstrap_features": [True, False],
                "classification__oob_score": [True, False],
                "classification__warm_start": [True, False],
            },
        }
    else:
        base_classifiers = [(name, clf) for name, clf in create_base_classifiers().items()]
        hyperparam_grids = [
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=MultinomialNB())],
                "classification__final_estimator__alpha": [0.1, 1, 10],
                "classification__final_estimator__fit_prior": [True, False],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=SVM(probability=True))],
                "classification__final_estimator__C": [0.1, 1, 10],
                "classification__final_estimator__kernel": ["linear", "rbf"],
                "classification__final_estimator__gamma": [0.001, 0.01, 0.1],
                "classification__final_estimator__degree": [2, 3, 4],
                "classification__final_estimator__probability": [True],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())],
                "classification__final_estimator__penalty": ["l1"],
                "classification__final_estimator__C": [0.1, 1, 10],
                "classification__final_estimator__solver": ["liblinear"],
                "classification__final_estimator__max_iter": [100, 200, 300],
                "classification__final_estimator__multi_class": ["auto", "ovr"],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())],
                "classification__final_estimator__penalty": ["l2"],
                "classification__final_estimator__C": [0.1, 1, 10],
                "classification__final_estimator__solver": ["liblinear", "lbfgs"],
                "classification__final_estimator__max_iter": [100, 200, 300],
                "classification__final_estimator__multi_class": ["auto", "ovr"],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=DecisionTreeClassifier())],
                "classification__final_estimator__max_depth": [3, 4, 5],
                "classification__final_estimator__min_samples_split": [2, 5, 10],
                "classification__final_estimator__min_samples_leaf": [1, 2, 4],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=RandomForestClassifier())],
                "classification__final_estimator__n_estimators": [100, 200, 300],
                "classification__final_estimator__criterion": ["gini", "entropy"],
                "classification__final_estimator__max_depth": [3, 4, 5],
                "classification__final_estimator__min_samples_split": [2, 5, 10],
                "classification__final_estimator__min_samples_leaf": [1, 2, 4],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
            {
                "classification": [StackingClassifier(estimators=base_classifiers, final_estimator=GradientBoostingClassifier())],
                "classification__final_estimator__n_estimators": [100, 200, 300],
                "classification__final_estimator__learning_rate": [0.1, 0.01, 0.001],
                "classification__final_estimator__max_depth": [3, 4, 5],
                "classification__final_estimator__subsample": [0.8, 0.9, 1.0],
                "classification__cv": [StratifiedKFold(n_splits=10, shuffle=True, random_state=42)],
            },
        ]
    return hyperparam_grids


def create_scoring():
    """
    Create a list of scoring metrics for evaluation.

    Returns:
        scoring (list): A list of scoring metrics, each represented as a tuple.
            The tuple contains the name of the metric and the corresponding scorer function.
    """
    scoring = [
        ("accuracy", make_scorer(accuracy_score)),
        ("precision", make_scorer(precision_score, average="macro")),
        ("recall", make_scorer(recall_score, average="macro")),
        ("f1", make_scorer(f1_score, average="macro")),
    ]
    return scoring


def perform_grid_search(
    pipeline, scoring, param_grid, X_train, y_train, random_state=42, nb_folds=10, return_best_classifier=False
):
    """
    Perform grid search to find the best hyperparameters for a given pipeline.

    Parameters:
        pipeline (object): The pipeline object to be optimized.
        scoring (str): The scoring metric used to evaluate the models.
        param_grid (dict): The parameter grid to search over.
        X_train (array-like): The training data.
        y_train (array-like): The target labels.
        random_state (int): The random state for reproducibility. Default is 42.
        nb_folds (int): The number of folds for cross-validation. Default is 10.

    Returns:
        best_parameters (dict): The best hyperparameters found by grid search.
        best_score (float): The best score obtained by grid search.
    """
    logger.info(f"Total (param_combinations, fits) to perform for grid search: {calculate_total_fits(param_grid, nb_folds)}")
    logger.info("Performing grid search...")
    cv = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=return_best_classifier,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_
    
    if return_best_classifier:
        best_estimator = grid_search.best_estimator_
        return best_parameters, best_score, best_estimator
    else:
        return best_parameters, best_score

def perform_grid_search_pipeline():
    pass

def calculate_metrics(y_value, y_pred):
    """
    Calculate evaluation metrics for a given scikit-learn pipeline.

    Parameters:
        y_value (array-like): The true labels for evaluation.
        y_pred (array-like): The predicted labels.

    Returns:
        accuracy (float): The accuracy score. The label predicted for a sample must exactly match the corresponding true label.
        precision (float): The precision score. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        recall (float): The recall score. The recall is intuitively the ability of the classifier to find all the positive samples.
        f1 (float): The F1 score.
        cm (array-like): The confusion matrix.
    """
    cm = confusion_matrix(y_value, y_pred)
    accuracy = accuracy_score(y_value, y_pred)
    precision = precision_score(y_value, y_pred, average="macro")
    recall = recall_score(y_value, y_pred, average="macro")
    f1 = f1_score(y_value, y_pred, average="macro")
    logger.info("Successfully extracted metrics.")
    return accuracy, precision, recall, f1, cm

def calculate_total_fits(param_grid, n_folds=10):
    """
    Calculate the total number of grid search fits for given parameter grids.
    
    :param parameter_grids: List of dictionaries where each dictionary represents
                            the parameter grid for a specific model.
    :param n_folds: The number of cross-validation folds (default is 5).
    :return: Total number of fits.
    """
    total_combinations = 0

    for grid in param_grid:
        # Calculate the number of combinations for each model's parameter grid
        combinations = len(param_grid)  # Initial value should be the amount of entries (classifiers) in the grid
        for values in grid.values():
            combinations *= len(values)
        total_combinations += combinations

    # Calculate total number of fits with cross-validation folds
    total_fits_with_folds = total_combinations * n_folds

    return total_combinations, total_fits_with_folds