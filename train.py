import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Step 1: Load Pretrained Model and Tokenizer
MODEL_NAME = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Step 2: Load Custom Chatbot Dataset 
dataset = load_dataset("csv", data_files={"train": "chatbot_data.csv"})

# Step 3: Preprocess Data (Tokenization)
def preprocess_function(examples):
    inputs = ["chatbot: " + text for text in examples["input"]]
    targets = [text for text in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./t5_chatbot",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)

# Step 5: Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)

trainer.train()

# Step 6: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_chatbot")
tokenizer.save_pretrained("./fine_tuned_chatbot")