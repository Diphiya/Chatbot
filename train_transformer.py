import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# Step 1: Define Intent Labels
intent_labels = {
    "greeting": 0,
    "goodbye": 1,
    "thanks": 2,
    "hours": 3,
    "password": 4
}

# Step 2: Create Training Data
training_data = [
    ("Hi!", "greeting"),
    ("Hello!", "greeting"),
    ("Good morning!", "greeting"),
    ("Bye!", "goodbye"),
    ("See you later!", "goodbye"),
    ("Thanks a lot!", "thanks"),
    ("What time are you open?", "hours"),
    ("I forgot my password.", "password"),
    ("How do I reset my password?", "password"),
]

# Step 3: Prepare Training Data
train_texts = [item[0] for item in training_data]
train_labels = [intent_labels[item[1]] for item in training_data]
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Step 4: Compute Class Weights
classes = np.array(list(intent_labels.values()))
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Step 5: Define a Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label,
        }

# Step 6: Initialize the Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("NerdyPy/fine_tuned_model_sentiment_analysis")
model = AutoModelForSequenceClassification.from_pretrained(
    "NerdyPy/fine_tuned_model_sentiment_analysis",
    num_labels=len(intent_labels),
    ignore_mismatched_sizes=True  # Handle mismatched weights
)

# Step 7: Modify Loss Function to Use Class Weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Step 8: Define a Custom Trainer Class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Accept **kwargs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Step 9: Create Train and Validation Datasets
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)

val_texts = ["How can I reset my password?", "What are your opening hours?", "Thanks!"]
val_labels = [intent_labels["password"], intent_labels["hours"], intent_labels["thanks"]]
val_labels = torch.tensor(val_labels, dtype=torch.long)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

# Step 10: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",  # Replaces deprecated evaluation_strategy
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    eval_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Step 11: Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 12: Train the Model
trainer.train()

# Step 13: Evaluate the Model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Step 14: Save the Model and Tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
print("Model and tokenizer have been saved to './model'!")

# Step 15: Load and Test the Model
loaded_model = AutoModelForSequenceClassification.from_pretrained("./model")
loaded_tokenizer = AutoTokenizer.from_pretrained("./model")

example_sentence = "I forgot my password!"
inputs = loaded_tokenizer(example_sentence, return_tensors="pt", truncation=True, padding=True)
outputs = loaded_model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
predicted_intent = list(intent_labels.keys())[predicted_class]

print(f"Predicted intent: {predicted_intent}")
