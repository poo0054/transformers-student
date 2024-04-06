from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.",
                          return_tensors="pt")
print(encoded_input)
print(encoded_input["input_ids"])

print(tokenizer.decode(encoded_input["input_ids"][0]))

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
# 将 padding 参数设置为 True 以填充批次中较短的序列以匹配最长的序列：
# 将 truncation 参数设置为 True 以将序列截断为模型接受的最大长度：+
# 将 return_tensors 参数设置为 pt （对于 PyTorch）或 tf （对于 TensorFlow）：
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_inputs)

print(tokenizer.decode(encoded_inputs["input_ids"][1]))
