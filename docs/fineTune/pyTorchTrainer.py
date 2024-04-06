# 首先加载 Yelp 评论数据集：
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

dataset = load_dataset("yelp_review_full")
# 正如您现在所知，您需要一个分词器来处理文本，并包含填充和截断策略来处理任何可变序列长度。要一步处理数据集，请使用 🤗 Datasets map 方法对整个数据集应用预处理函数：

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 首先加载模型并指定预期标签的数量。从 Yelp Review 数据集卡中，您知道有五个标签：


model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

# 指定保存训练检查点的位置：
training_args = TrainingArguments(output_dir="E:/cache/test_trainer")

# Trainer 在训练期间不会自动评估模型性能。您需要向 Trainer 传递一个函数来计算和报告指标。
# 🤗 Evaluate 库提供了一个简单的 accuracy 函数，您可以使用 evaluate.load （请参阅此快速教程了解更多信息）函数加载：


metric = evaluate.load("accuracy")


# 对 metric 调用 compute 以计算您的预测的准确性。在将预测传递给 compute 之前，您需要将 logits 转换为预测（记住所有 🤗 Transformers 模型都返回 logits）：
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 如果您想在微调期间监控评估指标，请在训练参数中指定 evaluation_strategy 参数，以在每个周期结束时报告评估指标：


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# 使用您的模型、训练参数、训练和测试数据集以及评估函数创建一个 Trainer 对象：
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
