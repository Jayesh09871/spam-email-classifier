# Spam Email Classifier

A spam email classifier using fine-tuned DistilBERT on Hugging Face, with a Streamlit UI for easy interaction.

## Project Structure

```
spam email classifier/
├── Dataset/                # Original dataset (place your emails.csv here)
├── data/                   # Dataset for Google Colab training
├── models/                 # Trained model output
├── src/
│   ├── __init__.py
│   ├── utils.py            # Utility functions
│   ├── preprocess.py       # Data loading and preprocessing
│   ├── train.py            # Training script
│   └── predict.py          # Prediction module
├── app.py                  # Streamlit app
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Training on Google Colab (Recommended)

1. Upload this repository to Google Drive or clone it in Colab
2. Upload your `emails.csv` to the `data/` directory
3. Open `src/train.py` and run the training script
4. After training, the model will be saved in `models/spam-email-classifier`

### 3. Uploading Model to Hugging Face Hub

1. Create an account on [Hugging Face Hub](https://huggingface.co/)
2. Install `huggingface_hub`
3. Run:
   ```python
   from huggingface_hub import login, push_to_hub
   login()
   push_to_hub(
       "models/spam-email-classifier",
       repo_id="your-username/spam-email-classifier",
       commit_message="Initial model upload"
   )
   ```

### 4. Running the Streamlit App Locally

Set your model path in `.env` file:

```env
MODEL_PATH=your-username/spam-email-classifier
```

Then run:
```bash
streamlit run app.py
```

## Environment Variables

Create a `.env` file with:
```env
MODEL_PATH=your-huggingface-model-name-or-local-path
```

## Notes

- The current `emails.csv` uses word frequency counts. For best results with DistilBERT, use a dataset with **raw email text** (columns: `text`, `label` where label is 0 for ham, 1 for spam)
- You can use datasets like [Enron Spam](https://www.kaggle.com/wanderfj/enron-spam) or [SpamAssassin](https://www.kaggle.com/venky73/spam-mails-dataset)
