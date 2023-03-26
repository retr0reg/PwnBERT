import os
import generate_code_segments.main as gc
import tokenize_codes.main as tc
import train.main as trains
from loguru import logger

def get_generate_results(amount):
    return gc.generate_codes(amount,True)
    
def generate_tokens(amount):
    codes_data = gc.generate_codes(amount,internal=True)
    tokenized_token = []

    size_v = gc.byte_to_kilobyte(os.path.getsize("generate_code_segments/vuln"))
    size_nv = gc.byte_to_kilobyte(os.path.getsize("generate_code_segments/nvuln"))
    logger.success(f"Size of vuln: {size_v}")
    logger.success(f"Size of non-vuln: {size_nv}")


def train():
    """ All requirements have been set in train/main.py """
    trains.main() 


def main():
    generate_tokens(20)
    
if __name__ == "__main__":
    main()