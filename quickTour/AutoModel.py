from torch import nn
from transformers import AutoModelForSequenceClassification
# 自动分词器
from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(pt_batch)

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

pt_outputs = pt_model(**pt_batch)
print(pt_outputs)

pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

pt_save_directory = "E:/cache/model/nlptown/bert-base-multilingual-uncased-sentiment-01"
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)
