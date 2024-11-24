import nltk
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer


nltk.download('punkt')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

# Load the tokenizer for the fine-tuned transformer model
tokenizer = BertTokenizer.from_pretrained("NerdyPy/fine_tuned_model_sentiment_analysis")
def preprocess_text(text):
    """
    Preprocesses input text for compatibility with the transformer model by tokenizing and lemmatizing.
    """
    # Basic NLP preprocessing: lowercase, tokenize, and lemmatize
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    preprocessed_text = " ".join(tokens)
    
    # Return the preprocessed text
    return preprocessed_text
