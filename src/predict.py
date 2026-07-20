from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from typing import Dict, Any
from src.utils import setup_logger

logger = setup_logger(__name__)

class SpamClassifier:
    def __init__(self, model_name_or_path: str):
        logger.info(f"Loading model from: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None  # Replace return_all_scores=True for new transformers versions
        )
    
    def predict(self, text: str) -> Dict[str, Any]:
        logger.info(f"Predicting for text of length: {len(text)}")
        results = self.classifier(text)[0]
        
        # Get the predicted label with highest score
        predicted_label = max(results, key=lambda x: x["score"])
        confidence = predicted_label["score"] * 100
        
        # Create probabilities dict
        probabilities = {r["label"]: r["score"] for r in results}
        
        return {
            "predicted_label": predicted_label["label"],
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "is_spam": predicted_label["label"] == "spam"
        }

if __name__ == "__main__":
    # Example usage
    classifier = SpamClassifier("Jayesh0987/spam-email-classifier")  # Replace with your model
    test_text = "Congratulations! You've won a free iPhone. Click here to claim your prize!"
    result = classifier.predict(test_text)
    print(result)
