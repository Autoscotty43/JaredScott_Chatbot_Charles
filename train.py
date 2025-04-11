from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from chatbot import clean_response
import os

MODEL_NAME = "google/flan-t5-small"

def preprocess(examples, tokenizer):
    inputs = ["Charles: " + text for text in examples["input"]]
    targets = [clean_response(text) for text in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

def train():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    dataset = load_dataset("csv", data_files={"train": "chatbot_data.csv"})
    tokenized = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)

    # Use a timestamp in the output directory name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"./trained_chatbot_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        save_total_limit=2,
        save_steps=500,
        warmup_steps=100,
        learning_rate=5e-5,
        report_to="none",  # Disable default logging to WandB/TensorBoard
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Trained model saved to {output_dir}") #important

if __name__ == "__main__":
    import datetime
    train()