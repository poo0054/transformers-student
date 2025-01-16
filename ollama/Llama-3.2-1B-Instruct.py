import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "meta-llama/Llama-3.2-1B-Instruct"

sys = """
    You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]
"""

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
messages = [
    {"role": "system",
     "content": sys},
    {"role": "user",
     "content": "Can you retrieve the details for the user with the ID 7890, who has black as their special request?"},
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
print("response------------\n", tokenizer.decode(response, skip_special_tokens=True))
