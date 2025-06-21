from openai import OpenAI
from config import Parameters
import os

client = OpenAI(
    api_key=Parameters.InternLM_KEY,  
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

chat_rsp = client.chat.completions.create(
     model="internlm3-latest",
     messages=[{
            "role": "user",         #role 支持 user/assistant/system/tool
            "content": "你知道刘慈欣吗？"
    }, {
            "role": "assistant",
            "content": "为一个人工智能助手，我知道刘慈欣。他是一位著名的中国科幻小说家和工程师，曾经获得过多项奖项，包括雨果奖、星云奖等。"
    },{
            "role": "user",
            "content": "他什么作品得过雨果奖？"
    }],
    stream=False
)

for choice in chat_rsp.choices:
    print(choice.message.content)
#若使用流式调用：stream=True，则使用下面这段代码
#for chunk in chat_rsp:
#    print(chunk.choices[0].delta.content)