import openai
from tqdm import tqdm
import config
from loguru import logger
import concurrent.futures


openai.api_key = config.OPEN_AI_KEY 

def setting():
    # logger.add(
    #     sys.stderr,
    #     format="<blue>[+]</blue> {message}",
    #     level="INFO"
    # )
    pass

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
    prompt = open("./prompt.txt", 'r').read()
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
                generated_code = generated_code.replace("code_ends", "")
                generated_code = "#include" + generated_code.split("#include")[-1]
                codes.append(generated_code)
                logger.success("Success!")
            except:
                logger.error("Some error occurred.")

    return codes
        
def write_given_data(data):
    try:
        f = open("./output.txt",'w+')
        if type(data) == list:
            for i in data:
                f.write(i)
        else:
            f.write(data)
    except:
        logger.error("Some error occured.")
        
    return 1

def generate_codes(code_generated_amount):
    setting()
    res = collect_generated_code(code_generated_amount)
    return write_given_data(res)    

if __name__=="__main__":
    generate_codes(5)