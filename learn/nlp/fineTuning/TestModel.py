import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")

# checkpoint = "bert-base-uncased"
# checkpoint = "./test-trainer/checkpoint-1000"
checkpoint = "./bert-base-uncased-text"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

index = 4

print(raw_datasets["train"][index])
print(raw_datasets["train"].features)
batch_encoding = tokenizer(raw_datasets["train"][index]["sentence1"], raw_datasets["train"][index]["sentence2"],
                           padding=True, truncation=True, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# model.config.id2label = {0: "not_equivalent", 1: "equivalent"}
print(batch_encoding)
output = model(**batch_encoding)

print(output)
print("softmax:", torch.nn.functional.softmax(output.logits, dim=-1))
print(model.config.id2label)
