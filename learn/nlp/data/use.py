from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="/home/poo0054/文档/data/SQuAD_it-train.json", field="data")
print(squad_it_dataset)
print(squad_it_dataset["train"][0])

# 很好, 我们已经加载了我们的第一个本地数据集! 但是, 虽然这对训练集有效, 但是我们真正想要的是包括 train 和 test 的 DatasetDict 对象。
# 这样的话就可以使用 Dataset.map() 函数同时处理训练集和测试集。
# 为此, 我们提供参数data_files的字典,将每个分割名称映射到与该分割相关联的文件：
data_files = {"train": "/home/poo0054/文档/data/SQuAD_it-train.json",
              "test": "/home/poo0054/文档/data/SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)
