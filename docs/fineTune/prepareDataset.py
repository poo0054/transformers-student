# 首先加载 Yelp 评论数据集：
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
print(dataset)
print(dataset["train"][100])

# 正如您现在所知，您需要一个分词器来处理文本，并包含填充和截断策略来处理任何可变序列长度。要一步处理数据集，请使用 🤗 Datasets map 方法对整个数据集应用预处理函数：
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
