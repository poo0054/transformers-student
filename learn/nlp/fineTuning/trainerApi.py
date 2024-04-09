from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 在我们定义我们的 Trainer 之前首先要定义一个 TrainingArguments 类，它将包含 Trainer用于训练和评估的所有超参数。
# 您唯一必须提供的参数是保存训练模型的目录，以及训练过程中的检查点。对于其余的参数，您可以保留默认值，这对于基本微调应该非常有效。
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

# 第二步是定义我们的模型。正如在[之前的章节](/2_Using Transformers/Introduction)一样，我们将使用 AutoModelForSequenceClassification 类，它有两个参数：

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 评估
# 让我们看看如何构建一个有用的 compute_metrics() 函数并在我们下次训练时使用它。
# 该函数必须采用 EvalPrediction 对象（带有 predictions 和 label_ids 字段的参数元组）并将返回一个字符串到浮点数的字典（字符串是返回的指标的名称，而浮点数是它们的值）。
# 我们可以使用 Trainer.predict() 命令来使用我们的模型进行预测：

predictions = trainer.predict(tokenized_datasets["validation"])
# predict() 的输出结果是具有三个字段的命名元组： predictions , label_ids ， 和 metrics .这 metrics 字段将只包含传递的数据集的 loss ，
# 以及一些运行时间（预测所需的总时间和平均时间）。如果我们定义了自己的 compute_metrics() 函数并将其传递给 Trainer ，该字段还将包含 compute_metrics()的结果。
print(predictions.predictions.shape, predictions.label_ids.shape, predictions.metrics)

# predict() 方法是具有三个字段的命名元组： predictions , label_ids ， 和 metrics .这 metrics 字段将只包含传递的数据集的 loss，以及一些运行时间（预测所需的总时间和平均时间）。
# 如果我们定义了自己的 compute_metrics() 函数并将其传递给 Trainer ，该字段还将包含compute_metrics() 的结果。
# 如你看到的， predictions 是一个形状为 408 x 2 的二维数组（408 是我们使用的数据集中元素的数量）。
# 这些是我们传递给predict()的数据集的每个元素的结果( logits )（正如你在之前的章节看到的情况）。要将我们的预测的可以与真正的标签进行比较，我们需要在第二个轴上取最大值的索引：
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

# 现在建立我们的 compute_metric() 函数来较为直观地评估模型的好坏，我们将使用 🤗 Evaluate 库中的指标。
# 我们可以像加载数据集一样轻松加载与 MRPC 数据集关联的指标，这次使用 evaluate.load() 函数。
# 返回的对象有一个 compute()方法我们可以用来进行度量计算的方法：
import evaluate

metric = evaluate.load("glue", "mrpc")
compute = metric.compute(predictions=preds, references=predictions.label_ids)
print(compute)

pt_save_directory = "D:/cache/model/demo1"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)
