from transformers import BertForSequenceClassification, BertTokenizer
import torch
import sys
import random
import generate_code_segments.main as gc

# 加载模型和标记器
model = BertForSequenceClassification.from_pretrained("./pwnbert_finetuned")
tokenizer = BertTokenizer.from_pretrained("./pwnbert_finetuned")

# def predict_vulnerability(model, tokenizer, code):
#     inputs = tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=-1)
#     label = torch.argmax(probabilities).item()

#     return label

# 准备要测试的文本数据
n=0
for i in range(int(sys.argv[1])):
    typies = ['nvuln','vuln']
    ram = random.choice(typies)
    with open(f'generate_code_segments/eval/{ram}/program_{i}.c', 'r') as f:
                text = f.read()
                # print(text)
    text = input("::: ")
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    # 使用模型输出进行预测或评估
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    label = torch.argmax(probabilities).item()
    if typies[label] == ram:
        print("Correct")
        print("****" + typies[label])
        n+=1
        
    else:
        print("Wrong")
        print("****" + typies[label])
        n+=0
        
print(f"Accuracy: {(n/int(sys.argv[1]))*100}%")
