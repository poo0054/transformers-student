from datasets import load_dataset
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose
from transformers import AutoImageProcessor

dataset = load_dataset("food101", split="train[:100]")
# 显示图片

print(dataset[0]["image"])

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 首先，让我们添加一些图像增强。您可以使用您喜欢的任何库，但在本教程中，我们将使用 torchvision 的 transforms 模块。
# 如果您有兴趣使用其他数据增强库，请在 Albumentations 或 Kornia 笔记本中了解如何操作。

# 1 在这里，我们使用 Compose 将几个转换链接在一起 - RandomResizedCrop 和 ColorJitter 。请注意，为了调整大小，我们可以从 image_processor 获取图像大小要求。
# 对于某些模型，需要精确的高度和宽度，而对于其他模型，仅定义 shortest_edge 。
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])


# 2 该模型接受 pixel_values 作为其输入。 ImageProcessor 可以负责图像的标准化，并生成适当的张量。
# 创建一个函数，将一批图像的图像增强和图像预处理结合起来并生成 pixel_values ：

def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples


# 3 然后使用 🤗 数据集 set_transform 动态应用转换：
dataset.set_transform(transforms)

# 4 现在，当您访问图像时，您会注意到图像处理器已添加 pixel_values 。您现在可以将处理后的数据集传递给模型！
dataset[0].keys()

import matplotlib.pyplot as plt

# 这是应用变换后图像的样子。图像已被随机裁剪，其颜色属性有所不同。
img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
