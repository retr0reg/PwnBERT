import time
import openai
from tqdm import tqdm
from . import config
from loguru import logger
import concurrent.futures
import os
from tenacity import retry, stop_after_attempt, wait_fixed


openai.api_key = config.OPEN_AI_KEY 

current_dir = os.path.dirname(os.path.abspath(__file__))

def setting():
    # logger.add(
    #     sys.stderr,
    #     format="<blue>[+]</blue> {message}",
    #     level="INFO"
    # )
    pass

def byte_to_kilobyte(b):
    return '%.1f' % float(b/1000) + 'KB'

def get_file_location(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
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

def process_code(future):
    try:
        respone = future.result()
        generated_code = respone.choices[0].message.content
        generated_code = generated_code.replace("code_start", "")
        generated_code = generated_code.replace("code_end", "")
        generated_code = "#include" + generated_code.split("#include")[-1]
        return generate_codes
    except Exception as e:
        logger.error(f"{e}; Retrying")
        process_code(future)

def collect_generated_code(amount_of_time):
    """This generates both vulnerable and non vulnerable code segments"""
    prompt4exist = open(get_file_location("prompt_for_exist.txt"), 'r').read()
    prompt4non = open(get_file_location("prompt_for_non_exist.txt"), 'r').read()
    logger.info("Process started")
    codes_exist = ''
    codes_non = ''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_async, prompt4exist) for _ in range(amount_of_time)]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            logger.info(f"Working on {i+1}th, for exist sample.")
            out = str(process_code(future))
            write_given_data(out,location="vuln/outputs.txt")
        
        futures = [executor.submit(get_async, prompt4non) for _ in range(amount_of_time)]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            logger.info(f"Working on {i+1}th, for non exist sample.")
            out = str(process_code(future))
            write_given_data(out,location="nvuln/outputs.txt")

    return codes_exist, codes_non
        
def write_given_data(datas,location=0,internal=False):
    try:
        if internal:
            return datas
        else:
            f = open(get_file_location(location),'a+')
            if type(datas) == list:
                f.write(str(datas))
                return 1
            else:
                f.write(datas)
                return 1
    except Exception as e:
        logger.error(f"ERROR: {e}")
        
    return 1

def generate_codes(code_generated_amount,internal):
    setting()
    res = collect_generated_code(code_generated_amount)
    return write_given_data(res,internal=internal)    

if __name__=="__main__":
    # generate_codes(1)
    pass