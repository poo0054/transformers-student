from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print("raw_datasets", raw_datasets)

raw_train_dataset = raw_datasets["train"]
print("simplate", raw_train_dataset[50])
print("features", raw_train_dataset.features)

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print("tokenize_function", tokenized_datasets)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]

samples = {
    k: v
    for k, v in samples.items()
    if k not in ["idx", "sentence1", "sentence2"]
}

batch = data_collator(samples)

print({k: v.shape for k, v in batch.items()})

print([len(x) for x in samples["input_ids"]])

print([len(x) for x in batch["input_ids"]])
