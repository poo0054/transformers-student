from transformers import pipeline

classifier = pipeline("sentiment-analysis")

l = classifier(["我今天非常开心", "我今天非常不开心"])

print(l)
