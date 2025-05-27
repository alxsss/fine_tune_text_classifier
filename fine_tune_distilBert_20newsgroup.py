import email
import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, get_scheduler
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import os

os.environ["WANDB_DISABLED"] = "true"

# Preprocessing the raw email data
def preprocess_newsgroup_row(data):
    # Extract only the subject and body from the email message
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"

    # Strip any remaining email addresses using a regex pattern
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)

    # Truncate the text to ensure it doesn't exceed the input length limit for the model
    text = text[:10000] 

    return text

# Preprocess the entire dataset
def preprocess_newsgroup_data(newsgroup_dataset):
    # Put data points into dataframe (data and target)
    df = pd.DataFrame(
        {"Text": newsgroup_dataset.data, "Label": newsgroup_dataset.target}
    )

    # Clean up the text by applying the row-level preprocessing function
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row)

    # Match label to target name index for better interpretability
    df["Class Name"] = df["Label"].map(lambda l: newsgroup_dataset.target_names[l])

    return df

newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# View list of class names for dataset

df_train = preprocess_newsgroup_data(newsgroups_train)
df_test = preprocess_newsgroup_data(newsgroups_test)

# convert df to dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)


checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
train_encodings = tokenizer(train_dataset[0]["Text"])

def tokenize_function(example):
    return tokenizer(example["Text"], truncation=True)


train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_tokenized_datasets = train_tokenized_datasets.remove_columns(["Text", "Class Name"])
train_tokenized_datasets = train_tokenized_datasets.rename_column("Label", "labels")

test_tokenized_datasets = test_tokenized_datasets.remove_columns(["Text", "Class Name"])
test_tokenized_datasets = test_tokenized_datasets.rename_column("Label", "labels")

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    test_tokenized_datasets, batch_size=8, collate_fn=data_collator
)
from transformers import TrainingArguments

training_args = TrainingArguments("Fine-tune-DistilBert")

#use accelerator
accelerator = Accelerator()
label_names = df_train['Class Name'].unique()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=len(label_names))
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.save_pretrained("./fine_tuned_model_20K_accelarator")
tokenizer.save_pretrained("./fine_tuned_model_20K_accelarator")

#evaluate model

# Assuming `test_dataset` is already tokenized and available
test_dl = eval_dataloader

# Set the model to evaluation mode
model.eval()

# Initialize variables to track the accuracy
correct = 0
total = 0
eval_loss = 0

# Evaluate on the test data
with torch.no_grad():
    for batch in test_dl:
        # Move data to the correct device
        batch = {key: value.to(model.device) for key, value in batch.items()}
        
        # Get model's predictions
        outputs = model(**batch)
        logits = outputs.logits
        loss = outputs.loss

        eval_loss += loss.item()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

# Calculate average loss and accuracy
avg_loss = eval_loss / len(test_dl)
accuracy = correct / total

print(f"Evaluation Loss: {avg_loss}")
print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")