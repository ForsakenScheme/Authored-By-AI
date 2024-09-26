from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.

from backend.scripts.detect_origin_web import detect_origin


def home(request):
    """
    View function for the home page.

    Retrieves user input and selected model, detects the origin of the input text,
    and returns predictions in JSON format.
    """
    if request.method == "POST":

        # Retrieve user input and selected model
        input_text = request.POST.get("user_input")
        selected_value = request.POST.get("selected_value")

        if selected_value is None:
            selected_value = "Stacking Support Vector Machine"

        # Detect origin of the input text
        try:
            predictions = detect_origin(input_text, selected_value)
        except Exception as e:
            return JsonResponse({"error": str(e)})

        # Convert predictions to a JSON-serializable format
        predictions_json = []
        for label, probabilities in predictions:
            prediction_dict = {"label": label, "probabilities": probabilities.tolist()}
            predictions_json.append(prediction_dict)

        # Return JSON response with result
        return JsonResponse({"predictions": predictions_json})
    else:
        # Render the homepage template
        return render(request, "home.html")
