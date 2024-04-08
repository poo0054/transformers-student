# 在上一个练习中，您看到了序列如何转换为数字列表。让我们将此数字列表转换为张量，并将其发送到模型：
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]

tokenizer1 = tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")

print(tokenizer1)
print(tokenizer1["input_ids"])
print(tokenizer1["input_ids"][0].shape)
print(tokenizer1["input_ids"][1].shape)
