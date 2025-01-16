import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "meta-llama/Llama-3.2-1B-Instruct"

sys = """
切割知识 日期：2023 年 12 月 今天日期：2024 年 7 月 23 日 当您收到工具调用响应时，使用输出格式化原始用户问题的答案。
您是一个具有工具调用功能的有用助手。

给定以下函数，请使用函数调用的 JSON 进行响应，并使用最能回答给定提示的正确参数。

以{"name"：function，"parameters"：参数名称及其值的字典} 的格式响应。不要使用变量，不存在的值不需要返回。
返回的格式例如：
{
  "name": "get_current_conditions",
  "parameters": {
    "location": "娄底市",
    "unit": "Celsius"
  }
}。

格式如下:
{
  "type": "function",
  "function": {
    "name": "get_current_conditions",
    "description": "获取特定位置的当前天气状况",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "市，例如娄底市，深圳市。从用户的位置推断这一点。"
        },
        "unit": {
          "type": "string",
          "enum": [
            "Celsius",
            "Fahrenheit"
          ],
          "description": "要使用的温度单位。从用户的位置推断这一点。"
        }
      },
      "required": [
        "location",
        "unit"
      ]
    }
  }
}
"""

user = """
北京天气怎么样
"""

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
messages = [
    {"role": "system",
     "content": sys},
    {"role": "user",
     "content": user}
]

print("messages======", tokenizer.apply_chat_template(messages,
                                                      tokenize=False,
                                                      add_generation_prompt=True, ))
print("messages======")

inputs = tokenizer.apply_chat_template(messages,
                                       return_tensors="pt",
                                       tokenize=True,
                                       add_generation_prompt=True,
                                       ).to(device)

# 生成文本
outputs = model.generate(
    inputs,
    do_sample=True,
    num_return_sequences=1
)

# 解码输出
decoded = tokenizer.batch_decode(outputs)

print("\nGenerated Text:")
print(decoded)

response = outputs[0][inputs.shape[-1]:]
print("response------------\n", tokenizer.decode(response))
