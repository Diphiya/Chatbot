

from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np  

# Intent labels and training data
intent_labels = {
    "greeting": 0,
    "goodbye": 1,
    "thanks": 2,
    "hours": 3,
    "password": 4
}

training_data = [
    ("Hi there!", "greeting"),
    ("Bye!", "goodbye"),
    ("Thanks a lot!", "thanks"),
    ("What are your opening hours?", "hours"),
    ("How can I reset my password?", "password"),
    ("I forgot my password. Can you help?", "password"),
    ("What time are you open?", "hours"),
    ("See you later!", "goodbye"),
    ("What time do you open?", "hours"),
    ("Can you tell me your business hours?", "hours"),
    ("What time does the store open and close?", "hours"),
    ("I need to reset my password.", "password"),
    ("Can you help me with a password reset?", "password"),
    ("I forgot my password. What should I do?", "password"),
    ("Tell me how to reset my password.", "password"),
    ("How do I reset the password for my account?", "password"),
]

# Convert labels to numerical format
train_texts = [item[0] for item in training_data]
train_labels = [intent_labels[item[1]] for item in training_data]

# Compute class weights based on label distribution
unique_labels = np.unique(train_labels)

# Compute the class weights for each class
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=unique_labels, 
    y=train_labels
)

# Convert class weights to a PyTorch tensor
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Print the class weights
print(f"Class weights: {class_weights}")

# Define chatbot_response function
def chatbot_response(user_input, tokenizer, model, data, intent_labels):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        response = "Hello! How can I assist you today?"
    elif "goodbye" in user_input or "bye" in user_input:
        response = "Goodbye! Have a great day! If you need anything, just ask."
    elif "thanks" in user_input or "thank you" in user_input:
        response = "You're welcome! Is there anything else I can help with?"
    elif "hours" in user_input or "time" in user_input:
        response = "Our opening hours are from 9 AM to 6 PM. Do you have any other questions?"
    elif "password" in user_input or "forgot" in user_input:
        response = ('To reset your password, visit the "Forgot Password" page on our website. '
                    'Would you like me to guide you to the page?')
    else:
        response = "I'm sorry, I didn't understand that. Can you please rephrase?"
    
    return response

if __name__ == "__main__":
    print("Chatbot: Hello! How can I assist you? (type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        
        response = chatbot_response(user_input, None, None, None, intent_labels)
        print(f"Chatbot: {response}")
