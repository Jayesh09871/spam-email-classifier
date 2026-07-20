import os
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def get_data_path() -> str:
    if os.path.exists("data/emails.csv"):
        return "data/emails.csv"
    elif os.path.exists("Dataset/emails.csv"):
        return "Dataset/emails.csv"
    else:
        raise FileNotFoundError("Dataset file not found! Check Dataset/ or data/ directories.")

def get_model_path() -> str:
    return "models/spam-email-classifier"

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
