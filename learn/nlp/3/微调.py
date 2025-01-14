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

training_args = TrainingArguments("ttt-trainer")

# 第二步是定义我们的模型。与上一章一样，我们将使用具有两个标签的 AutoModelForSequenceClassification 类：
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

# 一旦有了模型，我们就可以定义一个 Trainer，方法是将到目前为止构建的所有对象
# （模型、training_args、训练和验证数据集、data_collator 和 tokenizer）传递给它：
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
# 要在我们的数据集上微调模型，我们只需要调用 Trainer 的 train（） 方法：
trainer.train()

# 这将开始微调（在 GPU 上应该需要几分钟）并每 500 步报告一次训练损失。但是，它不会告诉您模型的性能如何（或差）。这是因为：
# 我们没有告诉 Trainer 在训练期间进行评估，而是将evaluation_strategy设置为“steps”（每 eval_steps评估一次）或 “epoch”（在每个 epoch 结束时评估）。
# 我们没有为 Trainer 提供 compute_metrics（） 函数来计算所述评估期间的指标（否则评估只会打印损失，这不是一个非常直观的数字）。

# 让我们看看如何构建一个有用的 compute_metrics（） 函数并在下次训练时使用它。
# 该函数必须采用 EvalPrediction 对象（该对象是具有 predictions 字段和 label_ids 字段的命名元组），
# 并将返回将字符串映射到浮点数的字典（字符串是返回的指标的名称，浮点数是其值）。
# 要从我们的模型中获得一些预测，我们可以使用 Trainer.predict（） 命令：
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
# predict（） 方法的输出是另一个命名元组，其中包含三个字段：predictions、label_ids 和 metrics。
# metrics 字段将仅包含所传递数据集的损失，以及一些时间指标（预测总时间和平均值）。
# 完成 compute_metrics（） 函数并将其传递给 Trainer 后，该字段还将包含 compute_metrics（） 返回的指标。


# 如您所见，predictions 是一个形状为 408 x 2 的二维数组（408 是我们使用的数据集中的元素数）。
# 这些是我们传递给 predict（） 的数据集的每个元素的 logits（正如您在上一章中看到的，所有 Transformer 模型都返回 logits）。
# 要将它们转换为我们可以与标签进行比较的预测，我们需要在第二个轴上获取具有最大值的索引：
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

# 我们现在可以将这些 preds 与标签进行比较。为了构建我们的 compute_metric（） 函数，我们将依赖 Evaluate 库中的🤗指标。
# 我们可以像加载数据集一样轻松地加载与 MRPC 数据集相关的指标，这次使用 evaluate.load（） 函数。
# 返回的对象有一个 table（） 方法，我们可以使用它来执行度量计算：

import evaluate

metric = evaluate.load("glue", "mrpc")
compute = metric.compute(predictions=preds, references=predictions.label_ids)
print("compute", compute)


# 您获得的确切结果可能会有所不同，因为模型头的随机初始化可能会改变它实现的指标。

# 在这里，我们可以看到我们的模型在验证集上的准确率为 85.78%，F1 分数为 89.97。
# 这是用于评估 GLUE 基准测试的 MRPC 数据集结果的两个指标。BERT 论文中的表格报告了基本模型的 F1 分数为 88.9。
# 这是我们目前使用有外壳模型时的无外壳模型，这解释了更好的结果。

# 将所有内容打包在一起，我们得到 compute_metrics（） 函数：
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
