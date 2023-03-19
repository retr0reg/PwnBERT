import sys
import openai
from tqdm import tqdm
import config
from loguru import logger

openai.api_key = config.OPEN_AI_KEY 

def setting():
    # logger.add(
    #     sys.stderr,
    #     format="<blue>[+]</blue> {message}",
    #     level="INFO"
    # )
    pass

def get(payload,amount):
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "user", 
                "content": str(payload),
            }
        ],
        n = int(amount)
    )
    #print(completion.choices[0].message.content)
    return completion

def collect_generated_code(amount_of_time):
    prompt = open("./prompt.txt",'r').read()
    logger.info("Process started")
    codes = []
    respone = get(prompt,amount_of_time)
    logger.info("Connection established")
    for i in tqdm(range(0,amount_of_time)):
        try:
            logger.info(f"working on {i}")
            generated_code = respone.choices[i].message.content
            generated_code = generated_code.replace("code_start", "")
            generated_code = generated_code.replace("code_end", "")
            generated_code = "#include" + generated_code.split("#include")[-1]
            codes.append(generated_code)
            logger.success("Sucess!")
            return codes
        except:
            logger.error("Some error happend.")
        

    

if __name__=="__main__":
    setting()
    print(collect_generated_code(2))