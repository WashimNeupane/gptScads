import os
from transformers import GPT2Tokenizer
from datasets import load_from_disk, load_dataset

# Load dataset
datasets = load_dataset("bigcode/the-stack-smol", data_dir="data/python")

# Take the first 95% of samples for training and the remaining 5% for validation
train_size = int(len(datasets['train']) * 0.95)

train_dataset = datasets['train'].select(range(train_size))
validation_dataset = datasets['train'].select(range(train_size, len(datasets['train'])))

# Initialize the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token
tokenizer.pad_token = tokenizer.eos_token

# Define the tokenize_function with truncation, padding, and max length
def tokenize_function(examples):
    return tokenizer(examples["content"], truncation=True, padding=True, max_length=1024)

# Apply tokenize_function to the train split
all_columns = list(train_dataset.features.keys())
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=all_columns
)

# Apply tokenize_function to the validation split
all_columns = list(validation_dataset.features.keys())
tokenized_validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=all_columns
)

# Define the block_size
#block_size = tokenizer.model_max_length
block_size = 256

# Define the group_texts function
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Apply group_texts function to the train split
lm_train_dataset = tokenized_train_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=1,
)

# Apply group_texts function to the validation split
lm_validation_dataset = tokenized_validation_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=1,
)

# Save preprocessed train and validation datasets to disk
lm_train_dataset.save_to_disk("data/processed_train_dataset")
lm_validation_dataset.save_to_disk("data/processed_validation_dataset")


