from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

print(config)

config = BertConfig()
model = BertModel(config)
print(config)
# Model is randomly initialized!


from transformers import BertModel

model = BertModel.from_pretrained("E:/cache/model/directory_on_my_computer")

model()
# model.save_pretrained("E:/cache/model/directory_on_my_computer")
