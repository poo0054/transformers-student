from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))
