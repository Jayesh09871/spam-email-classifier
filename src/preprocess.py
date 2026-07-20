import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Tuple, Dict
from src.utils import setup_logger, get_data_path

logger = setup_logger(__name__)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_and_prepare_dataset(
    text_col: str = None,
    label_col: str = "Prediction",
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[DatasetDict, Dict[int, str]]:
    logger.info("Loading dataset...")
    data_path = get_data_path()
    df = pd.read_csv(data_path)
    
    # Check if we have text column or word counts
    if text_col not in df.columns:
        # If no text column, assume word counts and create a dummy text
        # This is a fallback - ideally you should have raw email text
        logger.warning("No text column found. Creating dummy text from word frequencies.")
        word_cols = [col for col in df.columns if col not in ["Email No.", label_col]]
        df["text"] = df.apply(
            lambda row: " ".join([f"{col} " * int(row[col]) for col in word_cols if int(row[col]) > 0]),
            axis=1
        )
        text_col = "text"
    
    logger.info(f"Cleaning text in column: {text_col}")
    df[text_col] = df[text_col].apply(clean_text)
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(int)
    
    logger.info(f"Dataset shape after cleaning: {df.shape}")
    logger.info(f"Label distribution:\n{df[label_col].value_counts()}")
    
    # Split dataset
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val[label_col], random_state=42)
    
    logger.info(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    
    # Convert to Hugging Face DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train[[text_col, label_col]]),
        "val": Dataset.from_pandas(val[[text_col, label_col]]),
        "test": Dataset.from_pandas(test[[text_col, label_col]])
    })
    
    # Rename columns to standard names
    dataset_dict = dataset_dict.rename_columns({text_col: "text", label_col: "label"})
    
    # Create label mapping
    id2label = {0: "ham", 1: "spam"}
    label2id = {v: k for k, v in id2label.items()}
    
    return dataset_dict, id2label

def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 512
) -> Tuple[DatasetDict, AutoTokenizer]:
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    dataset_dict, id2label = load_and_prepare_dataset()
    tokenized_ds, tokenizer = tokenize_dataset(dataset_dict)
    print(tokenized_ds)
