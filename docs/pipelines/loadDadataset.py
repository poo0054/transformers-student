# 迭代数据集的最简单方法是从 🤗 数据集中加载一个数据集：
# KeyDataset is a util that will just output the item we're interested in.
from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
