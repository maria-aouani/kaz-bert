import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
os.environ["WANDB_MODE"] = "offline"


def main():
    # Load tokenizer and model
    MODEL_NAME = "amandyk/KazakhBERTmulti"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Load dataset from a text file
    dataset_path = "train.txt"
    try:
        dataset = load_dataset("text", data_files=dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./kazakh_bert_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="no",  # No evaluation dataset provided
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model("./kazakh_bert_finetuned")
    tokenizer.save_pretrained("./kazakh_bert_finetuned")

if __name__ == "__main__":
    main()
