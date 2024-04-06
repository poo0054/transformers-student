# 情感分析
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

l = classifier("We are very happy to show you the 🤗 Transformers library.")

print(l)

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
print(results)
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
