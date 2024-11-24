# Define intents and their corresponding labels
intent_labels = {
    "greeting": 0,
    "goodbye": 1,
    "thanks": 2,
    "hours": 3,
    "password": 4
}

# Dataset for responses
data = [
     {"intent": "greeting", "text": "Hello! How can I assist you today?"},
    {"intent": "goodbye", "text": "Goodbye! Have a great day! If you have more questions, feel free to come back."},
    {"intent": "thanks", "text": "You're welcome! Is there anything else I can help with?"},
    {"intent": "hours", "text": "We’re open from 9 AM to 6 PM."},
    {"intent": "password", "text": 'To reset your password, visit the "Forgot Password" page on our website.'},
]

# Training data for intent classification
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
    ("How do I recover my password?", "password"),
    ("I cannot log in because I forgot my password.", "password"),
    ("What is the process for resetting a password?", "password"),
    ("What time does your shop open?", "hours"),
    ("Can you tell me the opening and closing hours?", "hours"),
    ("When is your store open?", "hours"),
    ("What time do you close?", "hours"),
    ("Please let me know your business hours.", "hours"),
    ("What are your business hours?", "hours"),
    ("When do you open?", "hours"),
    ("What time are you open?", "hours"),
    ("Tell me your opening hours.", "hours"),
    ("How late do you stay open?", "hours"),
    ("Can you tell me the store hours?", "hours"),
    ("What time does your store close?", "hours"),


]
