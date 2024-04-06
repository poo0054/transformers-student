# 自动分词器
from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(pt_batch)
print(encoding)
# input_ids：令牌的数字表示。
# Attention_mask：指示应注意哪些标记。
