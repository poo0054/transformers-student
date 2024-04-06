# 这会在提供的 4 个音频文件上运行管道，但它会将它们以 2 个批次的形式传递给模型（位于 GPU 上，批处理更有可能提供帮助），
# 而不需要您提供任何进一步的代码。输出应该始终与您在没有批处理的情况下收到的输出相匹配。
from transformers import pipeline

transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto", batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)
print(texts)
