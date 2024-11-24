from model import tokenizer, model
from dataset import data, intent_labels
from chatbot_response import chatbot_response  
import numpy as np  
from sklearn.utils.class_weight import compute_class_weight



train_labels = [0, 1, 2, 3, 4]  

# Correct compute_class_weight call
# Generate class weights for imbalanced classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(list(intent_labels.values())),  # Convert values of intent_labels to numpy array
    y=train_labels
)

print(f"Class weights: {class_weights}")


if __name__ == "__main__":
    print("Chatbot: Hello! How can I assist you? (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input, tokenizer, model, data, intent_labels)
        print(f"Chatbot: {response}")
