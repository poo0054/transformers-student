# 基于词的(Word-based)
tokenized_text = "Jim Henson was a puppeteer".split()
print('split', tokenized_text)

# 基于字符(Character-based)


# 子词标记化
# 例如，“annoyingly”可能被认为是一个罕见的词，可以分解为“annoying”和“ly”。这两者都可能作为独立的子词出现得更频繁，同时“annoyingly”的含义由“annoying”和“ly”的复合含义保持。


# 标记化过程由标记器(tokenizer)的tokenize() 方法实现：

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print('tokenize', tokens)

# 从词符(token)到输入 ID
ids = tokenizer.convert_tokens_to_ids(tokens)

print('ids', ids)

# 解码(Decoding) 正好相反：从词汇索引中，我们想要得到一个字符串。这可以通过 decode() 方法实现，如下：
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
