import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import unittest
from unittest.mock import patch
from backend.scripts.database_functions import get_nb_of_processed_texts
from backend.scripts.pipelines import (
    UserConfigPipeline,
    get_all_possible_classifiers,
    create_stacking_classifiers,
    create_base_classifiers,
    create_final_estimators,
    create_bagging_classifiers,
)
from sklearn.ensemble import StackingClassifier, BaggingClassifier
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


class TestUserConfigPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = UserConfigPipeline()
        self.all_possible_classifiers = get_all_possible_classifiers()
        self.custom_mock_pipeline = None

    def test_init(self):
        self.assertIsNotNone(self.pipeline.X)
        self.assertIsNotNone(self.pipeline.y)
        self.assertEqual(len(self.pipeline.X), len(self.pipeline.y))
        self.assertGreater(len(self.pipeline.X), 0)
        self.assertGreater(len(self.pipeline.y), 0)
        self.assertEqual(len(self.pipeline.X), get_nb_of_processed_texts())
        self.assertEqual(set(self.pipeline.y), set(["ai", "human"]))
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.random_state)
        self.assertIsNotNone(self.pipeline.nb_folds)
        self.assertGreaterEqual(
            int(self.pipeline.config.get("RandomState", "random_state")), 0
        )
        if self.pipeline.config.getboolean("Preprocessing", "stemm") == True:
            self.assertFalse(
                self.pipeline.config.getboolean("Preprocessing", "lemmatize")
            )
        if self.pipeline.config.getboolean("Preprocessing", "lemmatize") == True:
            self.assertFalse(self.pipeline.config.getboolean("Preprocessing", "stemm"))
        self.assertTrue(
            self.pipeline.config.get("FeatureSelection", "nb_features").isdigit()
            or self.pipeline.config.get("FeatureSelection", "nb_features") == "all"
        )
        self.assertGreater(
            int(self.pipeline.config.get("FeatureSelection", "nb_features")),
            0 or self.pipeline.config.get("FeatureSelection", "nb_features") == "all",
        )
        self.assertGreaterEqual(self.pipeline.nb_folds, 2)
        self.assertIsNone(self.pipeline.classifier_name)
        self.assertIsNotNone(self.pipeline.X_train)
        self.assertIsNotNone(self.pipeline.X_val)
        self.assertIsNotNone(self.pipeline.X_test)
        self.assertIsNotNone(self.pipeline.y_train)
        self.assertIsNotNone(self.pipeline.y_val)
        self.assertIsNotNone(self.pipeline.y_test)
        self.assertIsNone(self.pipeline.custom_pipeline)
        self.assertIsNotNone(self.pipeline.param_grids)
        self.assertIsNone(self.pipeline.param_grid)

    def test_setClassifierName(self):
        for classifier in self.all_possible_classifiers:
            self.pipeline.set_classifier_name(classifier)
            self.assertEqual(self.pipeline.classifier_name, classifier)

    def test_setCustomPipeline(self):
        for classifier in self.all_possible_classifiers:
            self.pipeline.set_classifier_name(classifier)
            self.pipeline.set_custom_pipeline()
            self.assertIsNotNone(self.pipeline.custom_pipeline)

    def test_setParamGrid(self):
        for classifier in get_all_possible_classifiers():
            self.pipeline.set_classifier_name(classifier)
            self.pipeline.set_param_grid(classifier)
            self.assertIsNotNone(self.pipeline.param_grid)

    @patch("backend.scripts.pipelines.load")
    def test_loadCustomPipeline(self, mock_load):
        mock_load.return_value = "custom_pipeline"
        self.pipeline.load_custom_pipeline("model.joblib")
        self.assertEqual(self.pipeline.custom_pipeline, "custom_pipeline")

    def test_create_custom_pipeline(self):
        for classifier in self.all_possible_classifiers:
            self.pipeline.set_classifier_name(classifier)
            self.pipeline.set_custom_pipeline()
            self.pipeline.set_param_grid(classifier)
            self.pipeline.create_custom_pipeline()
            self.assertIsNotNone(self.pipeline.custom_pipeline)


class TestCreateBaseClassifiers(unittest.TestCase):
    def test_create_base_classifiers(self):
        base_classifiers_dict = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Support Vector Machine": SVM(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }
        base_classifiers = create_base_classifiers()
        self.assertIsInstance(base_classifiers, dict)
        self.assertGreater(len(base_classifiers), 0)
        for classifier_name, classifier in base_classifiers.items():
            self.assertIsInstance(
                classifier, base_classifiers_dict[classifier_name].__class__
            )
            self.assertTrue(hasattr(classifier, "fit"))
            self.assertTrue(hasattr(classifier, "predict"))
            self.assertTrue(hasattr(classifier, "predict_proba"))
            self.assertTrue(hasattr(classifier, "score"))


class TestCreateFinalEstimators(unittest.TestCase):
    def test_create_final_estimators(self):
        estimators_dict = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Support Vector Machine": SVM(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }
        final_estimators = create_final_estimators()
        self.assertIsInstance(final_estimators, dict)
        self.assertGreater(len(final_estimators), 0)
        for estimator_name, estimator in final_estimators.items():
            self.assertIsInstance(estimator, estimators_dict[estimator_name].__class__)
            self.assertTrue(hasattr(estimator, "fit"))
            self.assertTrue(hasattr(estimator, "predict"))
            self.assertTrue(hasattr(estimator, "predict_proba"))
            self.assertTrue(hasattr(estimator, "score"))


class TestCreateStackingClassifiers(unittest.TestCase):
    def test_create_stacking_classifiers(self):
        base_classifiers = create_base_classifiers()
        final_estimators = create_final_estimators()
        stacking_classifiers = create_stacking_classifiers(
            base_classifiers, final_estimators
        )
        self.assertIsInstance(stacking_classifiers, dict)
        self.assertEqual(len(stacking_classifiers), len(final_estimators))
        for estimator_name, stacking_classifier in stacking_classifiers.items():
            self.assertTrue(estimator_name.startswith("Stacking "))
            self.assertIsInstance(stacking_classifier, StackingClassifier)
            self.assertEqual(
                stacking_classifier.estimators, list(base_classifiers.items())
            )
            test_estimator = " ".join(estimator_name.split(" ")[1:])
            self.assertEqual(
                stacking_classifier.final_estimator, final_estimators[test_estimator]
            )


class TestCreateBaggingClassifiers(unittest.TestCase):
    def test_create_bagging_classifiers(self):
        base_classifiers = create_base_classifiers()
        final_estimators = create_final_estimators()
        bagging_classifiers = create_bagging_classifiers(
            base_classifiers, final_estimators
        )
        self.assertIsInstance(bagging_classifiers, dict)
        self.assertEqual(len(bagging_classifiers), len(final_estimators))
        for estimator_name, bagging_classifier in bagging_classifiers.items():
            self.assertTrue(estimator_name.startswith("Bagging "))
            self.assertIsInstance(bagging_classifier, BaggingClassifier)
            test_estimator = " ".join(estimator_name.split(" ")[1:])
            self.assertEqual(
                bagging_classifier.estimator, base_classifiers[test_estimator]
            )


if __name__ == "__main__":
    unittest.main()
