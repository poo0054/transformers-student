# 该管道还可以对大型数据集运行推理。我们建议执行此操作的最简单方法是使用迭代器：
from transformers import pipeline


def data():
    for i in range(10):
        yield f"My example {i}"


pipe = pipeline(model="openai-community/gpt2", device_map="auto")
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])

print(generated_characters)
