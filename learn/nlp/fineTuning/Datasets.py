from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
# 在上面的例子之中,Label（标签） 是一种ClassLabel（分类标签），使用整数建立起到类别标签的映射关系。0对应于not_equivalent，1对应于equivalent。
print(raw_train_dataset.features)
# 为了预处理数据集，我们需要将文本转换为模型能够理解的数字。正如你在第二章上看到的那样
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )


#
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# 请注意，我们现在在标记函数中省略了padding参数。这是因为在标记的时候将所有样本填充到最大长度的效率不高。
# 一个更好的做法：在构建批处理时填充样本更好，因为这样我们只需要填充到该批处理中的最大长度，而不是整个数据集的最大长度。
# 当输入长度变化很大时，这可以节省大量时间和处理能力!
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# 负责在批处理中将数据整理为一个batch的函数称为collate函数。它是你可以在构建DataLoader时传递的一个参数，
# 默认是一个函数，它将把你的数据集转换为PyTorch张量，并将它们拼接起来(如果你的元素是列表、元组或字典，则会使用递归)。
# 这在我们的这个例子中下是不可行的，因为我们的输入不是都是相同大小的。
# 我们故意在之后每个batch上进行填充，避免有太多填充的过长的输入。
# 这将大大加快训练速度，但请注意，如果你在TPU上训练，这可能会导致问题——TPU喜欢固定的形状，即使这需要额外的填充。
from transformers import DataCollatorWithPadding

# 为了解决句子长度统一的问题，我们必须定义一个collate函数，该函数会将每个batch句子填充到正确的长度
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[print(len(x)) for x in samples["input_ids"]]

batch = data_collator(samples)

var = {k: v.shape for k, v in batch.items()}

print(var)
