import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_scheduler
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

# Load the dataset
df = pd.read_csv("train_kazakh_2_corrected.csv")

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a custom Dataset class
class TextToTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_length=512, max_output_length=512):
        self.inputs = dataframe["input"].tolist()
        self.targets = dataframe["target"].tolist()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        # Tokenize input and target
        input_ids = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        labels = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        return {"input_ids": input_ids, "labels": labels}

# Create the dataset and DataLoader
dataset = TextToTextDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Set up the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = len(dataloader) * num_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Mixed precision training setup
scaler = GradScaler()

# Training loop
model.train()
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Mixed precision training
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})

print("Training complete!")

# Inference example
example_2 = "Менің ауа рай ерте жақсы болады."

# Tokenize the input and move to the appropriate device
inputs = tokenizer(example_2, return_tensors="pt").to(device)

# Generate output
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode and print the generated text
result_2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:", result_2)
