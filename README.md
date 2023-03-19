# PwnBERT
A project based on BERT to detect GLIBC vulnerabilities.
## What is PwnBERT
PwnBERT is a BERT-based vulnerability detection tool designed to identify and analyze Pwn-related vulnerabilities (e.g. UAF, heap overflow, etc.) in C language. By combining natural language processing techniques and security domain knowledge, this project aims to provide an efficient and reliable solution to help developers and security researchers identify potential security risks and thus strengthen code security.

## Where are we currently?
Currently, we are still in the development stage of the project. However, We had decided to use `BERT` as ours’s Pretrained Module, furthermore use `CodeBERT`
as assistance.

### What is CodeBERT
CodeBERT is a state-of-the-art neural model for code representation learning. It is based on the Transformer architecture and is pre-trained on a large corpus of code. CodeBERT can be fine-tuned on various downstream tasks such as code classification, code retrieval, and code generation. By leveraging the pre-trained model, CodeBERT can effectively capture the semantic and syntactic information of code, which makes it a powerful tool for code analysis and understanding. In PwnBERT, we use CodeBERT to assist in identifying and analyzing Pwn-related vulnerabilities in C language.

## How to use?
In our project, we generally seperated our plan into few parts;
1: Making the trainset 
    - using elaborately designed prompt to generate specific codes section
    - data marking (TODO LIST)