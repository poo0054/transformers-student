from transformers import pipeline

pipe = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v2", device_map="auto",
                return_timestamps=True)

result = pipe("E:/poo00/Documents/录音/录音.mp3")

print(result)
