#  加载模型
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

pt_save_directory = "E:/cache/model/nlptown/bert-base-multilingual-uncased-sentiment-01"
pt_model = AutoModelForSequenceClassification.from_pretrained(pt_save_directory)

tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(pt_batch)

pt_outputs = pt_model(**pt_batch)
pt_predictions = nn.functional.softmax(input=pt_outputs.logits, dim=-1)
print(pt_predictions)
# get max tensor

# 获取最大值
max_val, max_index = pt_predictions.max(dim=1)

# 输出最大值
print(f"最大值是: {max_val.item()}，对应的索引是: {max_index.item()}")
