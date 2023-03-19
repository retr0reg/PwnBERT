import openai
from tqdm import tqdm

openai.api_key = ""

def get(payload):
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "user", 
                "content": str(payload),
            }
        ],
        presence_penalty= 2.0,
        temperature=1.5
        # n = 5
    )
    print(completion.choices[0].message.content)
    return completion

def collect_generated_code(amount_of_time):
    prompt = open("./prompt.txt",'r').read()
    codes = []
    for _ in tqdm(range(amount_of_time)):
        respone = get(prompt)
        generated_code = respone.choices[0].text
        generated_code = generated_code.replace("code_start", "")
        generated_code = generated_code.replace("code_end", "")
        codes.append(generated_code)
    return codes

    

if __name__=="__main__":
    print(collect_generated_code(10))