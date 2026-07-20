import streamlit as st
from src.predict import SpamClassifier
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")
st.markdown("Classify emails as **Spam** or **Ham** (Not Spam) using a fine-tuned DistilBERT model.")

# Model selection
MODEL_PATH = os.getenv("MODEL_PATH", "distilbert-base-uncased")

# Load model
@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    return SpamClassifier(MODEL_PATH)

try:
    classifier = load_classifier()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Please check your MODEL_PATH environment variable or Hugging Face Hub model name.")
    st.stop()

# Input area
st.subheader("Enter Email Content")
email_content = st.text_area(
    "Paste your email text here:",
    height=200,
    placeholder="Dear user, you have won a prize! Click here to claim..."
)

if st.button("Classify Email", type="primary"):
    if email_content.strip() == "":
        st.warning("Please enter some email content to classify!")
    else:
        with st.spinner("Classifying..."):
            result = classifier.predict(email_content)
        
        # Display result
        st.subheader("Classification Result")
        
        if result["is_spam"]:
            st.error(f"⚠️ This email is **Spam** (Confidence: {result['confidence']}%)")
        else:
            st.success(f"✅ This email is **Ham** (Not Spam) (Confidence: {result['confidence']}%)")
        
        # Show probabilities
        with st.expander("View Probabilities"):
            st.write(f"- Spam: {result['probabilities'].get('spam', 0)*100:.2f}%")
            st.write(f"- Ham: {result['probabilities'].get('ham', 0)*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Hugging Face Transformers and Streamlit")
