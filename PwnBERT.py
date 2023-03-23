import generate_code_segments.main as gc
import tokenize_codes.main as tc
from loguru import logger

def get_generate_results(amount):
    return gc.generate_codes(amount,True)
    
def generate_tokens(amount):
    codes_data = gc.generate_codes(amount,internal=True)
    tokenized_token = []
    for i in codes_data[0]:
        tokenized_token.append(tc.encode_tokenizer(str(i)))
    gc.write_given_data(tokenized_token,location="vuln/outputs.txt")
    tokenized_token = []
    for i in codes_data[1]:
        tokenized_token.append(tc.encode_tokenizer(str(i)))
    gc.write_given_data(tokenized_token,location="nvuln/outputs.txt")
    


def main():
    generate_tokens(2)
    
if __name__ == "__main__":
    main()