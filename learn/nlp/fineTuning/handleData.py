from datasets import load_dataset
from transformers import AutoTokenizer

# Datasets库提供了一个非常便捷的命令，可以在模型中心（hub）上下载和缓存数据集。我们可以通过以下的代码下载MRPC数据集：
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

# 我们可以访问我们数据集中的每一个raw_train_dataset对象，如使用字典：
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

# 我们可以看到标签已经是整数了，所以我们不需要对标签做任何预处理。
# 要知道哪个数字对应于哪个标签，我们可以查看raw_train_dataset的features. 这将告诉我们每列的类型：
# 在上面的例子之中,Label（标签） 是一种ClassLabel（分类标签），使用整数建立起到类别标签的映射关系。0对应于not_equivalent，1对应于equivalent。
print(raw_train_dataset.features)

# 为了预处理数据集，我们需要将文本转换为模型能够理解的数字。正如你在第二章上看到的那样
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

b = tokenizer(raw_datasets["train"][15]["sentence1"], raw_datasets["train"][15]["sentence2"])
# 我们在第二章 讨论了输入词id(input_ids) 和 注意力遮罩(attention_mask) ，但我们在那个时候没有讨论类型标记ID(token_type_ids)。
# 在这个例子中，类型标记ID(token_type_ids)的作用就是告诉模型输入的哪一部分是第一句，哪一部分是第二句。
print(b)

# 现在我们已经了解了标记器如何处理一对句子，我们可以使用它对整个数据集进行处理：如之前的章节，我们可以给标记器提供一组句子，第一个参数是它第一个句子的列表，第二个参数是第二个句子的列表。
# 这也与我们在第二章中看到的填充和截断选项兼容. 因此，预处理训练数据集的一种方法是：
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)


# 这很有效，但它的缺点是返回字典（字典的键是输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 类型标记ID(token_type_ids)，字典的值是键所对应值的列表）。
# 而且只有当您在转换过程中有足够的内存来存储整个数据集时才不会出错（而🤗数据集库中的数据集是以Apache Arrow文件存储在磁盘上，因此您只需将接下来要用的数据加载在内存中
# ，因此会对内存容量的需求要低一些）。


# 为了将数据保存为数据集，我们将使用Dataset.map()方法，如果我们需要做更多的预处理而不仅仅是标记化，那么这也给了我们一些额外的自定义的方法。
# 这个方法的工作原理是在数据集的每个元素上应用一个函数，因此让我们定义一个标记输入的函数：
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# 此函数的输入是一个字典（与数据集的项类似），并返回一个包含输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 类型标记ID(token_type_ids) 键的新字典。
# 请注意，如果像上面的示例一样，如果键所对应的值包含多个句子（每个键作为一个句子列表），那么它依然可以工作，就像前面的例子一样标记器可以处理成对的句子列表。
# 这样的话我们可以在调用map()使用该选项 batched=True ，这将显著加快标记与标记的速度。这个标记器来自🤗 Tokenizers库由Rust编写而成。当我们一次给它大量的输入时，这个标记器可以非常快。
# 请注意，我们现在在标记函数中省略了padding参数。这是因为在标记的时候将所有样本填充到最大长度的效率不高。
# 一个更好的做法：在构建批处理时填充样本更好，因为这样我们只需要填充到该批处理中的最大长度，而不是整个数据集的最大长度。当输入长度变化很大时，这可以节省大量时间和处理能力!
# 下面是我们如何在所有数据集上同时应用标记函数。我们在调用map时使用了batch =True，这样函数就可以同时应用到数据集的多个元素上，而不是分别应用到每个元素上。这将使我们的预处理快许多

# 下面是我们如何在所有数据集上同时应用标记函数。我们在调用map时使用了batch =True，这样函数就可以同时应用到数据集的多个元素上，而不是分别应用到每个元素上。这将使我们的预处理快许多
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print(tokenized_datasets)

# 在使用预处理函数map()时，甚至可以通过传递num_proc参数使用并行处理。
# 我们在这里没有这样做，因为🤗标记器库已经使用多个线程来更快地标记我们的样本，但是如果您没有使用该库支持的快速标记器，使用num_proc可能会加快预处理。
# 我们的标记函数(tokenize_function)返回包含输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 类型标记ID(token_type_ids) 键的字典,
# 所以这三个字段被添加到数据集的标记的结果中。注意，如果预处理函数map()为现有键返回一个新值，那将会修改原有键的值。
# 最后一件我们需要做的事情是，当我们一起批处理元素时，将所有示例填充到最长元素的长度——我们称之为动态填充。


# 为了解决句子长度统一的问题，我们必须定义一个collate函数，该函数会将每个batch句子填充到正确的长度。幸运的是，🤗transformer库通过DataCollatorWithPadding为我们提供了这样一个函数。
# 当你实例化它时，需要一个标记器(用来知道使用哪个词来填充，以及模型期望填充在左边还是右边)，并将做你需要的一切:
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
