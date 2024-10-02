import os
import logging
import datetime


def setup_logging(application_type):
    """
    Singleton function to setup logging for the application
    """
    log_dir = ""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if application_type == "local":
        log_dir = "code/backend/logs/"
    
    elif application_type == "web":
        log_dir = "code/django_abai/abai_website/logs/"
    else:
        raise ValueError("Invalid application type for logging, only web and local are supported.")    
    log_file = os.path.join(log_dir, f"{application_type}-{current_time}.log")

    # Ensure the directory exists, create it if not
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger()
