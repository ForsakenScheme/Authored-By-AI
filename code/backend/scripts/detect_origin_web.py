import numpy as np

from joblib import load
from pathlib import Path
from sklearn.pipeline import Pipeline

def detect_origin(texts: list, data_language: str = "english", metric: str = "accuracy") -> str:
    """
    Detects the origin of the input text using the specified model.

    Parameters:
        texts (list): The input text to detect the origin of.
        data_language (str): The language of the data used to train the model. Default is "english".
        metric (str): The metric used to determine which model to load. Default is "accuracy".

    Returns:
        predictions (list[tuple(predicted_label, predicted_probability)]): The list containing predicted label and predicted probability for each of the input texts.
    """
    # Construct the directory path to look for .joblib files
    model_directory = Path(__file__).resolve().parent.parent.parent / f"backend/models/{data_language}/best/{metric}"

    try:
        # Find the first .joblib file in the directory
        joblib_files = list(model_directory.glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(f"No model files found for metric '{metric}' in '{model_directory}'.")

        # Load the first .joblib file found
        model_pipeline = load(joblib_files[0])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The model file for metric '{metric}' was not found. It probably hasn't been trained yet.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")

    # If text contains a newline character, split the text into a list of strings and remove empty strings
    if isinstance(texts, str):  
        texts = texts.split("\n")  
    texts = [t for t in texts if t.strip() != ""]  

    predictions = predict_unknown_texts(model_pipeline, texts)
    for prediction in predictions:
        assert (
            prediction[0] == "ai"
            or prediction[0] == "human"
            or prediction[0] == "skipped (too short)"
        ), "Invalid prediction"
    return predictions



def predict_unknown_texts(model_pipeline: Pipeline, texts, min_length=50):
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
    for text in texts:
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
