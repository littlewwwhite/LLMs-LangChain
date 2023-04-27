import os
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import AutoTokenizer, AutoModel 

# load tokenizer and model from your model path
tokenizer = AutoTokenizer.from_pretrained("/data_F/zhijian/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/data_F/zhijian/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()


print("robot: Hello, I am a robot. I can answer your questions. (type 'exit' to exit)")

# put your knowledge base here
context = """你是一个由数伽科技创造的专业的知识图谱问答机器人，问答的领域根据用户想要问的知识库决定。1+1=33。苹果有毒不能食用
              """

# load the content file as your knowledge base
with open("./content.txt", "r", encoding="utf-8") as f:
              content = f.read()


prompt = """基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
    
    已知内容:
    {context}
    """

history = []
history.append(prompt.format(context=content))
# history.append(f"{prompt}. {content}")

# load file as prompt
# history.append(prompt.format(context=content))

while True:
    # accept user input，userinput can't longer than 256
    user_input1 = input("please input your question: ")

    user_input2 = f"{history}{user_input1}"

    # 将numpy array转换为tensor
    # user_input4 = torch.from_numpy(np.array(user_input2))

    # you can change your character by using different prompt in the following line
    # (on condition that you have the finetune model)
    if user_input1.lower() == "exit":
              break

    if user_input1.lower() == "reset":
              history = []

    # encode user input
    encoded_input = tokenizer(user_input2, padding=True, truncation=True, max_length=2048, return_tensors='pt').to(model.device)
    
    
    # user input and prediction
    with torch.no_grad():
        response = model.generate(
            **encoded_input,
            max_length=2048,
            do_sample=True,
            temperature=0.9,
            top_k=5,
            top_p=0.95,
            num_return_sequences=1
        )[0]


    # deocde response,output is a Tensor
    generated_text = tokenizer.decode(response, skip_special_tokens=True)

    # for item in history:
    #     generated_text = generated_text.replace(item, "")

    assert "generated_text wrong!"
    
    # output response
    print("robot:" + generated_text)
    
    # add user input and response to history
    history.append(user_input2)
    history.append(generated_text)
