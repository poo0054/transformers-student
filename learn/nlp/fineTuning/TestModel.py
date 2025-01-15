import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")

# checkpoint = "bert-base-uncased"
checkpoint = "./test-trainer/checkpoint-1000"
# checkpoint = "./bert-base-uncased-text"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

index = 4

print(raw_datasets["train"][index])
print(raw_datasets["train"].features)
# batch_encoding = tokenizer(raw_datasets["train"][index]["sentence1"], raw_datasets["train"][index]["sentence2"],
#                            padding=True, truncation=True, return_tensors="pt")
batch_encoding = tokenizer(
    'The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .',
    'PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .',
    padding=True, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# model.config.id2label = {0: "not_equivalent", 1: "equivalent"}
output = model(**batch_encoding)

print("output------", output)
print("softmax------", torch.nn.functional.softmax(output.logits, dim=-1))
print("model------", model.config.id2label)
# softmax: tensor([[0.0039, 0.9961]], grad_fn=<SoftmaxBackward0>)
# The stock rose $ 2.11 , or about 11 percent , to close Friday at $ 21.51 on the New York Stock Exchange .
# 该股周五在纽约证券交易所收于 21.51 点，上涨 2.11 点，即约 11%。
# PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .
# 周五，PG&E Corp. 股价在纽约证券交易所上涨 1.63 美元，即 8%，至 21.03 美元。
