from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer

# 1 您将从 PreTrainedModel 或 torch.nn.Module 开始：
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

# 2 TrainingArguments 包含您可以更改的模型超参数，例如学习率、批量大小和训练周期数。如果您不指定任何训练参数，则使用默认值：
training_args = TrainingArguments(
    output_dir="E:/cache/model/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)
# 3 加载预处理类，例如分词器、图像处理器、特征提取器或处理器：
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# 4 加载数据集：
dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT


# 5 创建一个函数来标记数据集：
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


#  然后使用地图将其应用到整个数据集：
dataset = dataset.map(tokenize_dataset, batched=True)

# 6 DataCollatorWithPadding 用于从数据集中创建一批示例：
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 现在将所有这些类收集到 Trainer 中：

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

# 准备好后，调用 train() 开始训练：
trainer.train()
