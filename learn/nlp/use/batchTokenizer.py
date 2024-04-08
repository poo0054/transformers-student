# 在上一个练习中，您看到了序列如何转换为数字列表。让我们将此数字列表转换为张量，并将其发送到模型：
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)
# This line will fail.
output = model(input_ids)
print("output:", output)
print("Logits:", output.logits)
