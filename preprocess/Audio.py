from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

print(dataset[0]["audio"])

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

print(dataset[0]["audio"]["array"])

# 使用 AutoFeatureExtractor.from_pretrained() 加载特征提取器：
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

audio_input = [dataset[0]["audio"]["array"]]
feature_extractor(audio_input, sampling_rate=16000)

print(dataset[0]["audio"]["array"].shape)

print(dataset[1]["audio"]["array"].shape)


# 创建一个函数来预处理数据集，使音频样本具有相同的长度。指定最大样本长度，特征提取器将填充或截断序列以匹配它：
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )
    return inputs


# 将 preprocess_function 应用于数据集中的前几个示例：
print(preprocess_function(dataset[:5])["input_values"][0].shape.shape)

# 将 preprocess_function 应用于数据集中的前几个示例：
processed_dataset = preprocess_function(dataset[:5])
print(processed_dataset["input_values"][0].shape)

print(processed_dataset["input_values"][1].shape)
