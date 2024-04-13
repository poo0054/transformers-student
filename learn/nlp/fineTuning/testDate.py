from datasets import load_dataset
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


raw_datasets = load_dataset("glue", "mrpc")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print(tokenized_datasets)
print(tokenized_datasets["train"].features)
print(tokenized_datasets["train"][0])
