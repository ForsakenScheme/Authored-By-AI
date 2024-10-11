import logging
import os
from datetime import datetime

def setup_logging():
    """
    Set up logging with a timestamped log file for the entire application.
    """
    # Define the log directory
    log_dir = os.getcwd() + "/code/backend/logs"

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamped log file name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{current_time}.log")

    # Set up basic configuration for logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_logger(name=None):
    """
    Retrieve a logger with the specified name.
    
    Args:
        name (str): Name of the logger, usually __name__ of the calling module.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)

