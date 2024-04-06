from transformers import pipeline

transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result)

transcriber1 = pipeline(model="openai/whisper-large-v2", chunk_length_s=30, return_timestamps=True)
l = transcriber1("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")
print(l)
