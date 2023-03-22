import openai
from tqdm import tqdm
from . import config
from loguru import logger
import concurrent.futures
import os


openai.api_key = config.OPEN_AI_KEY 

current_dir = os.path.dirname(os.path.abspath(__file__))

def setting():
    # logger.add(
    #     sys.stderr,
    #     format="<blue>[+]</blue> {message}",
    #     level="INFO"
    # )
    pass

def get_file_location(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

def get_async(payload):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": str(payload),
            }
        ],
    )
    return completion

def collect_generated_code(amount_of_time):
    prompt = open(get_file_location("prompt_for_exist.txt"), 'r').read()
    logger.info("Process started")
    codes = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_async, prompt) for _ in range(amount_of_time)]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            logger.info(f"Working on {i}")
            try:
                respone = future.result()
                generated_code = respone.choices[0].message.content
                generated_code = generated_code.replace("code_start", "")
                generated_code = generated_code.replace("code_end", "")
                generated_code = "#include" + generated_code.split("#include")[-1]
                codes.append(generated_code)
                logger.success("Success!")
            except:
                logger.error("Some error occurred.")

    return codes
        
def write_given_data(data,internal=False):
    try:
        if internal:
            return data
        else:
            f = open(get_file_location("output.txt"),'w+')
            if type(data) == list:
                f.write(str(data))
                return 1
            else:
                f.write(data)
                return 1
    except:
        logger.error("Some error occured.")
        
    return 1

def generate_codes(code_generated_amount,internal):
    setting()
    res = collect_generated_code(code_generated_amount)
    return write_given_data(res,internal=internal)    

if __name__=="__main__":
    generate_codes(1)