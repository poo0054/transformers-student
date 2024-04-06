# 模型属性是随机初始化的，您需要先训练模型，然后才能使用它来获得有意义的结果。
# 首先导入 AutoConfig，然后加载要修改的预训练模型。
# 在 AutoConfig.from_pretrained() 中，您可以指定要更改的属性，例如注意力头的数量：
# https://huggingface.co/docs/transformers/create_a_model
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)

my_model = AutoModel.from_config(my_config)
