from datasets import load_dataset

dataset = load_dataset("food101", split="train[:100]")

print(dataset[0]["image"])
