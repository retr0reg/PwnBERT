# PwnBERT
A project based on BERT to detect GLIBC vulnerabilities.
## What is PwnBERT
PwnBERT is a BERT-based vulnerability detection tool designed to identify and analyze Pwn-related vulnerabilities (e.g. UAF, heap overflow, etc.) in C language. By combining natural language processing techniques and security domain knowledge, this project aims to provide an efficient and reliable solution to help developers and security researchers identify potential security risks and thus strengthen code security.

## Where are we currently?
Currently, we are still in the development stage of the project. However, We had decided to use `BERT` as ours’s Pretrained Module, furthermore use `CodeBERT`
as assistance.

### What is CodeBERT
CodeBERT is a state-of-the-art neural model for code representation learning. It is based on the Transformer architecture and is pre-trained on a large corpus of code. CodeBERT can be fine-tuned on various downstream tasks such as code classification, code retrieval, and code generation. By leveraging the pre-trained model, CodeBERT can effectively capture the semantic and syntactic information of code, which makes it a powerful tool for code analysis and understanding. In PwnBERT, we use CodeBERT to assist in identifying and analyzing Pwn-related vulnerabilities in C language.

## Plan introduction
In our project, we generally seperated our plan into few parts;
* Making the trainset 
    * using elaborately designed prompt to generate specific codes section
    * data marking (TODO LIST)

## How to use?
### `generate_code_segments`
In this part, what we basically did is use `OpenAI API` 's `ChatGPT` to generate our prompt, then extract the code in `collect_generated_code(amount_of_time):`. You can test our code by following these steps:

1. `$ touch config.py` This will create a config file that will be used later for the `generate_code_segment.py` file

2. `$ echo "OPEN_AI_KEY = #YOUR_API_KEY"` Change `#YOUR_API_KEY` to your OpenAI API KEY

3. `$ python3 generate_code_segment` This will run the python file.

## Update Logs
**Mainly updates after Mar 22, 2023:**

* v1.1, Mar 20: Started to use `concurrent.futures` for acceleration purposes.

* v1.2, Mar 22: Created `PwnBERT.py`, major adjust the structure of directories (because I need to import them), fix minor bugs and added new stuff on `generate_code_segments/` and `tokenize_codes`.