import os
import logging
import datetime


def setup_logging(application_type):
    """
    Singleton function to setup logging for the application
    """
    log_dir = ""

    if application_type == "local":
        log_dir = "code/backend/logs"
    else:
        raise ValueError("Invalid application type")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if application_type == "local":
        log_file = os.path.join(log_dir, f"{application_type}-{current_time}.log")
    else:
        raise ValueError("Invalid application type")

    # Ensure the directory exists, create it if not
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger()
