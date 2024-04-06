# 1 首先创建 pipeline() 并指定推理任务：
from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition")

# 2 将您的输入传递给 pipeline()。在语音识别的情况下，这是一个音频输入文件：

# transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
result = transcriber(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)

print(result)

# 不是你想要的结果吗？查看 Hub 上下载次数最多的一些自动语音识别模型，看看是否可以获得更好的转录。
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v2")
l = transcriber([
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
])

print(l)
