# putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 第二步是定义我们的模型。
# 正如在[之前的章节](/2_Using Transformers/Introduction)一样，我们将使用 AutoModelForSequenceClassification 类，它有两个参数：
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.config.id2label = {0: "not_equivalent", 1: "equivalent"}


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


raw_datasets = load_dataset("glue", "mrpc")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 在我们定义我们的 Trainer 之前首先要定义一个 TrainingArguments 类，它将包含 Trainer用于训练和评估的所有超参数。
# 您唯一必须提供的参数是保存训练模型的目录，以及训练过程中的检查点。对于其余的参数，您可以保留默认值，这对于基本微调应该非常有效。

training_args = TrainingArguments("test-trainer")


# 最后将所有东西打包在一起，我们得到了我们的 compute_metrics() 函数：
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("glue", "mrpc")
    return metric.compute(predictions=predictions, references=labels)


# 一旦我们有了我们的模型，我们就可以定义一个 Trainer 通过将之前构造的所有对象传递给它——我们的model 、training_args ，
# 训练和验证数据集，data_collator ，和 tokenizer ：
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 预测
predictions = trainer.predict(tokenized_datasets["validation"])
print("predict------", predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

metric = evaluate.load("glue", "mrpc")
compute = metric.compute(predictions=preds, references=predictions.label_ids)

print("compute------", compute)

# 保存模型
tokenizer.save_pretrained("./bert-base-uncased-text1")
model.save_pretrained("./bert-base-uncased-text1")
