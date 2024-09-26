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

from backend.utils.log import setup_logging
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

logger = setup_logging("local")


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

        predict_unkown_texts(text): Predicts the label for unknown texts.
    """

    def __init__(self, logger=logger):
        # get all texts and labels from database.sqlite
        self.X, self.y = get_all_texts_and_labels_from_table("processed")
        assert len(self.X) == len(self.y)
        assert len(self.X) > 0
        assert len(self.y) > 0
        # assert value of len(X) is equal to the number of rows in the database
        assert len(self.X) == get_nb_of_processed_texts()
        # assert y contains both "ai" and "human" labels
        if len(set(self.y)) != 2:
            raise ValueError(
                "Be sure to have both 'ai' and 'human' labels in the database."
            )
        assert set(self.y) == set(["ai", "human"])
        # get user configuration from config.ini file
        self.config = config
        # retrieve random state for reproducibility
        self.random_state = self.config.getint("RandomState", "random_state")
        # retrieve number of folds from config.ini file
        self.nb_folds = self.config.getint("CrossValidation", "nb_folds")
        # assertions for the configuration file
        assert (
            int(self.config.get("RandomState", "random_state")) >= 0
            or self.config.get("RandomState", "random_state") == None
        ), "Random state must be a positive integer or None."
        if self.config.getboolean("Preprocessing", "stemm"):
            assert (
                self.config.getboolean("Preprocessing", "lemmatize") == False
            ), "Lemmatization must be disabled if stemming is enabled."
        if self.config.getboolean("Preprocessing", "lemmatize"):
            assert (
                self.config.getboolean("Preprocessing", "stemm") == False
            ), "Stemming must be disabled if lemmatization is enabled."
        assert (
            self.config.get("FeatureSelection", "nb_features").isdigit()
            or self.config.get("FeatureSelection", "nb_features") == "all"
        ), "Number of features must be a positive integer or 'all'."
        assert (
            int(self.config.get("FeatureSelection", "nb_features")) > 0
            or self.config.get("FeatureSelection", "nb_features") == "all"
        ), "Number of features must be a positive integer or 'all'."
        assert self.nb_folds >= 2, "Number of folds must be at least 2."
        # retrieve selected classifier name from config.ini file
        self.classifier_name = None
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (
            split_data(self.X, self.y)
        )
        self.custom_pipeline = None
        self.param_grids = create_all_hyperparam_grids()
        self.param_grid = None
        self.logger = logger
        self.logger.info("UserConfigPipeline initialized.")

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

    def setParamGrid(self, classifier_name):
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
        file_path = (
            Path(__file__).resolve().parent.parent.parent
            / "backend/models"
            / (model + ".joblib")
        )
        model_pipeline = load(file_path)
        self.custom_pipeline = model_pipeline

    def create_custom_pipeline(self):
        """
        Creates a custom pipeline based on the user's configuration.

        Returns:
            Pipeline: The custom pipeline.
        """
        self.logger.info("Creating custom pipeline...")
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
        }
        self.logger.info(f"pipeline config: {pipeline_config}")
        # create a pipeline with the specified config
        pipeline = create_pipeline(**pipeline_config)
        self.logger.info("Pipeline created.")
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
        elif self.classifier_name == "Logistic Regression":
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
        elif self.classifier_name == "Stacking Logistic Regression":
            classifier = StackingClassifier(
                estimators=list(create_base_classifiers().items()),
                final_estimator=LogisticRegression(),
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
        self.logger.info(f"Selected classifier: {self.classifier_name}")
        # create custom pipeline with feature extraction from Before and After preprocessing, also apply classifier and feature selection
        custom_pipeline = Pipeline(
            [("pipeline", pipeline), ("classification", classifier)]
        )
        self.logger.info("Custom pipeline created.")
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
        self.logger.info(
            f"\n{equal_signs_training}\n{title_training}\n{equal_signs_training}"
        )
        self.logger.info("Fitting model.")
        self.custom_pipeline = self.custom_pipeline.fit(self.X_train, self.y_train)
        self.logger.info("Successfully fitted.")

        if test:
            return self.custom_pipeline
        # save the trained model to a file
        with open(
            ("code/backend/models/" + self.classifier_name + ".joblib"), "wb"
        ) as joblib_file:
            dump(self.custom_pipeline, joblib_file)
            self.logger.info("Successfully dumped trained custom pipeline to file.")

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
        self.logger.info(
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
        self.logger.info("Successfully validated custom model...")
        self.logger.info(f"Validation accuracy: {validation_accuracy}")
        self.logger.info(f"Validation precision: {validation_precision}")
        self.logger.info(f"Validation recall: {validation_recall}")
        self.logger.info(f"Validation F1 score: {validation_f1}")
        self.logger.info(f"Validation confusion matrix:\n{validation_cm}")
        return validation_dict

    def grid_search(self):
        """
        Performs grid search to find the best hyperparameters for the custom model.

        Returns:
            best_parameters_for_score (list): A list of tuples containing the best parameters and scores for each scoring metric.
        """
        best_parameters_for_score = []
        for score in create_scoring():
            best_params, best_score = perform_grid_search(
                self.custom_pipeline,
                score[1],
                self.param_grid,
                self.X_train,
                self.y_train,
                self.random_state,
                self.nb_folds,
            )
            best_parameters_for_score.append(
                ((score[0], best_score), best_params)
            )  # example (("precision", 0.88)), {"classification__C": 1, "classification__kernel": "linear"})
        return best_parameters_for_score

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
        self.logger.info(f"\n{equal_signs}\n{title}\n{equal_signs}")
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
        self.logger.info("Successfully tested custom model...")
        self.logger.info(f"Test accuracy: {test_accuracy}")
        self.logger.info(f"Test precision: {test_precision}")
        self.logger.info(f"Test recall: {test_recall}")
        self.logger.info(f"Test F1 score: {test_f1}")
        self.logger.info(f"Test confusion matrix:\n{cm}")
        self.logger.info("Extracting test prediction probabilities...")
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
        "Logistic Regression": LogisticRegression(
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
        "Logistic Regression": LogisticRegression(
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

    Returns:
        sklearn.pipeline.Pipeline: The model pipeline.
    """
    logger.info("Creating pipeline...")
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
                        "afterPreprocessingPipeline",
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
    logger.info(f"Pipeline config: {steps}")
    return Pipeline(steps)


def create_all_hyperparam_grids():
    """
    Create hyperparameter grids for different classifiers.

    Returns:
        dict: A dictionary containing hyperparameter grids for different classifiers.
    """
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
        "Logistic Regression": {
            "classification__penalty": ["l1", "l2"],
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
        },
        "Stacking Support Vector Machine": {
            "classification__final_estimator__C": [0.1, 1, 10],
            "classification__final_estimator__kernel": ["linear", "rbf"],
            "classification__final_estimator__gamma": [0.001, 0.01, 0.1],
            "classification__final_estimator__degree": [2, 3, 4],
            "classification__final_estimator__probability": [True],
        },
        "Stacking Logistic Regression": {
            "classification__final_estimator__penalty": ["l1", "l2"],
            "classification__final_estimator__C": [0.1, 1, 10],
            "classification__final_estimator__solver": ["liblinear", "lbfgs"],
            "classification__final_estimator__max_iter": [100, 200, 300],
            "classification__final_estimator__multi_class": ["auto", "ovr"],
        },
        "Stacking Decision Tree": {
            "classification__final_estimator__max_depth": [3, 4, 5],
            "classification__final_estimator__min_samples_split": [2, 5, 10],
            "classification__final_estimator__min_samples_leaf": [1, 2, 4],
        },
        "Stacking Random Forest": {
            "classification__final_estimator__n_estimators": [100, 200, 300],
            "classification__final_estimator__criterion": ["gini", "entropy"],
            "classification__final_estimator__max_depth": [3, 4, 5],
            "classification__final_estimator__min_samples_split": [2, 5, 10],
            "classification__final_estimator__min_samples_leaf": [1, 2, 4],
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
    pipeline, scoring, param_grid, X_train, y_train, random_state=42, nb_folds=2
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
        nb_folds (int): The number of folds for cross-validation. Default is 2.

    Returns:
        best_parameters (dict): The best hyperparameters found by grid search.
        best_score (float): The best score obtained by grid search.
    """
    logger.info("Performing grid search...")
    cv = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=False,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_
    logger.info(f"Best parameters: {best_parameters}")
    logger.info(f"Best score: {best_score}")
    logger.info("Successfully performed grid search...")
    return best_parameters, best_score


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
    logger.info(
        f"\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nConfusion matrix:\n\t{cm}"
    )
    return accuracy, precision, recall, f1, cm
