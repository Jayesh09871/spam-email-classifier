import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from src.preprocess import load_and_prepare_dataset, tokenize_dataset
from src.utils import setup_logger, get_model_path, ensure_dir

logger = setup_logger(__name__)

def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    load_precision = evaluate.load("precision")
    load_recall = evaluate.load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    precision = load_precision.compute(predictions=predictions, references=labels)["precision"]
    recall = load_recall.compute(predictions=predictions, references=labels)["recall"]
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_model(
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512
):
    # Load and prepare data
    dataset_dict, id2label = load_and_prepare_dataset()
    tokenized_ds, tokenizer = tokenize_dataset(dataset_dict, model_name, max_length)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )
    
    # Set training arguments
    output_dir = get_model_path()
    ensure_dir(output_dir)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=10
    )
    
    # Train
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_ds["test"])
    logger.info(f"Test results: {test_results}")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    train_model()
