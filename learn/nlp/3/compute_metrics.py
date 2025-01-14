import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 定义 Trainer 之前的第一步是定义一个 TrainingArguments 类，该类将包含 Trainer 将用于训练和评估的所有超参数。
# 您必须提供的唯一参数是保存训练模型的目录，以及沿途的检查点。对于所有其他操作，您可以保留默认值，这应该可以很好地进行基本的微调。
from transformers import TrainingArguments

training_args = TrainingArguments("compute_metrics")

# 第二步是定义我们的模型。与上一章一样，我们将使用具有两个标签的 AutoModelForSequenceClassification 类：
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

import evaluate


# 将所有内容打包在一起，我们得到 compute_metrics（） 函数：
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
