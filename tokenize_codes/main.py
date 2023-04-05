from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def user_input_and_tokenize():
    user_input = input("Enter your input: ")
    return tokenizer.encode(user_input)

def encode_tokenizer(input):
    return tokenizer.encode(input)

def decode_tokenizer(encoded_input):
    return tokenizer.decode(encoded_input)

if __name__ == "__main__":
    user_input_and_tokenize()