#  分词器
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
print(result)
