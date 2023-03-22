import generate_code_segments.main as gc
import tokenize_codes.main as tc
from loguru import logger

def get_generate_results(amount):
    return gc.generate_codes(amount,True)
    
def main():
    tokenized_token = []
    res = get_generate_results(1)
    logger.info(res)
    for i in res:
        tokenized_token.append(tc.encode_tokenizer(str(res)))
    
    print(tokenized_token)
    
if __name__ == "__main__":
    main()