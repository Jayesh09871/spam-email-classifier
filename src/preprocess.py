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
    # Basic text cleaning
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def detect_text_column(df: pd.DataFrame) -> str:
    """Auto-detect the text column in the DataFrame."""
    text_candidates = ["text", "message", "email", "EmailText", "content", "body"]
    for col in text_candidates:
        if col in df.columns:
            return col
    # If no known candidate, pick the column with the longest average string length
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) == 0:
        raise ValueError("No text columns found in the dataset!")
    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in text_cols}
    return max(avg_lengths, key=avg_lengths.get)

def detect_label_column(df: pd.DataFrame) -> str:
    """Auto-detect the label column in the DataFrame."""
    label_candidates = ["spam", "label", "Category", "Prediction", "class"]
    for col in label_candidates:
        if col in df.columns:
            return col
    # If no known candidate, pick the column with 2-10 unique values
    for col in df.columns:
        if 2 <= df[col].nunique() <= 10:
            return col
    raise ValueError("No suitable label column found in the dataset!")

def encode_labels(labels: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
    """Encode labels to 0 (ham) and 1 (spam)."""
    unique_labels = sorted(labels.unique())
    # Try to map common label names
    label_map = {}
    for label in unique_labels:
        if isinstance(label, str):
            lower_label = label.lower()
            if lower_label in ["ham", "0", "not spam", "legitimate", "safe"]:
                label_map[label] = 0
            elif lower_label in ["spam", "1", "junk", "phishing"]:
                label_map[label] = 1
    # If not enough mappings, use index-based mapping
    if len(label_map) < 2:
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    # Create id2label
    id2label = {idx: "ham" if idx == 0 else "spam" for idx in label_map.values()}
    return labels.map(label_map), id2label

def load_and_prepare_dataset() -> Tuple[DatasetDict, Dict[int, str]]:
    logger.info("Loading dataset...")
    data_path = get_data_path()
    df = pd.read_csv(data_path)
    
    # Detect columns
    text_col = detect_text_column(df)
    label_col = detect_label_column(df)
    logger.info(f"Detected text column: {text_col}, label column: {label_col}")
    
    # Clean text
    logger.info("Cleaning text...")
    df[text_col] = df[text_col].apply(clean_text)
    
    # Encode labels
    logger.info("Encoding labels...")
    df[label_col], id2label = encode_labels(df[label_col])
    
    # Drop NaNs
    df = df.dropna(subset=[text_col, label_col])
    logger.info(f"Dataset shape after cleaning: {df.shape}")
    logger.info(f"Label distribution:\n{df[label_col].value_counts()}")
    
    # Split dataset
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1/(1-0.2), stratify=train_val[label_col], random_state=42)
    logger.info(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    
    # Convert to Hugging Face DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train[[text_col, label_col]]),
        "val": Dataset.from_pandas(val[[text_col, label_col]]),
        "test": Dataset.from_pandas(test[[text_col, label_col]])
    })
    
    # Rename columns to standard names
    dataset_dict = dataset_dict.rename_columns({text_col: "text", label_col: "label"})
    
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
    
    return tokenized_datasets, tokenizer

if __name__ == "__main__":
    dataset_dict, id2label = load_and_prepare_dataset()
    tokenized_ds, tokenizer = tokenize_dataset(dataset_dict)
    print("Preprocessing complete!")
