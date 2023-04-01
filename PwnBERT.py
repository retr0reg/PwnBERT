import os
import sys
import generate_code_segments.main as gc
import tokenize_codes.main as tc
import train.main as trains
from loguru import logger

def get_generate_results(amount):
    return gc.generate_codes(amount,True)
    
def generate_tokens(amount):
    codes_data = gc.generate_codes(amount,internal=True,both=True)
    tokenized_token = []

    size_v = gc.size_to_human(os.path.getsize("generate_code_segments/vuln/outputs.txt"))
    size_nv = gc.size_to_human(os.path.getsize("generate_code_segments/nvuln/outputs.txt"))
    logger.success(f"Size of vuln: {size_v}")
    logger.success(f"Size of non-vuln: {size_nv}")


def train():
    """ All requirements have been set in train/main.py """
    trains.main()


def main():
    generate_tokens(sys.argv[1])
    
if __name__ == "__main__":
    main()