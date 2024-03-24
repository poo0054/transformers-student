from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

audio_input = [dataset[0]["audio"]["array"]]

extractor = feature_extractor(audio_input, sampling_rate=16000)

print(extractor)


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


processed_dataset = preprocess_function(dataset[:5])

print(processed_dataset)
