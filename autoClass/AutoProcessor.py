# 多模式任务需要一个结合了两种类型预处理工具的处理器。
# 例如，LayoutLMV2模型需要一个图像处理器来处理图像，需要一个分词器来处理文本；处理器将两者结合在一起。
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
