from transformers import pipeline

generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to"))

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
l = generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2, )

print(l)
