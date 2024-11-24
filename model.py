from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("NerdyPy/fine_tuned_model_sentiment_analysis")
model = AutoModelForSequenceClassification.from_pretrained("NerdyPy/fine_tuned_model_sentiment_analysis", num_labels=3)


# Verify the model and tokenizer
if __name__ == "__main__":
    print("Model and tokenizer loaded successfully.")
