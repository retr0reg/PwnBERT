# PwnBERT
A project based on BERT to detect GLIBC vulnerabilities.

## What is PwnBERT
PwnBERT is a BERT-based vulnerability detection tool designed to identify and analyze Pwn-related vulnerabilities (e.g. UAF, heap overflow, etc.) in C language. By combining natural language processing techniques and security domain knowledge, this project aims to provide an efficient and reliable solution to help developers and security researchers identify potential security risks and thus strengthen code security.
### What is PwnBERT (In the way that human can understand)
Generally speaking, PwnBERT is a tool that helps find and analyze vulnerabilities in computer programs written in C language that could be exploited by attackers. It uses a technique called natural language processing and combines it with security expertise to make it easier for developers and security researchers to identify potential security risks and make code more secure. The goal is to create a reliable and efficient solution for identifying and preventing potential security threats.

Why you should Pay attention on this project?
* it used OpenAI API (ChatGPT) for acquiring training set
* it used AI training to detect complex vulns in codes, instead of identifying them via structure analyzing.

* it's made by me, ALL BY MY SELF! (i know it does not sound like a good reason but i will put it up here anyway:) )


## Where are we currently?
Currently, we have finished our main fine-tuning of our BERT module ( after all we still decided to use DistilBert :sadfaceemoji  ) However, the accuracy is still not ideable. WE ARE STILL WORKING ON THAT NOW

### What is CodeBERT
CodeBERT is a state-of-the-art neural model for code representation learning. It is based on the Transformer architecture and is pre-trained on a large corpus of code. CodeBERT can be fine-tuned on various downstream tasks such as code classification, code retrieval, and code generation. By leveraging the pre-trained model, CodeBERT can effectively capture the semantic and syntactic information of code, which makes it a powerful tool for code analysis and understanding. In PwnBERT, we use CodeBERT to assist in identifying and analyzing Pwn-related vulnerabilities in C language.

## Plan introduction
In our project, we generally seperated our plan into few parts;
* Making the trainset 
    * using elaborately designed prompt to generate specific codes section
    * data marking 
    * fine-tuning

## How to use?
### `generate_code_segments`
In this part, what we basically did is use `OpenAI API` 's `ChatGPT` to generate our prompt, then extract the code in `collect_generated_code(amount_of_time):`. You can test our code by following these steps:

1. `$ touch config.py` This will create a config file that will be used later for the `main.py` file

2. `$ echo "OPEN_AI_KEY = #YOUR_API_KEY"` Change `#YOUR_API_KEY` to your OpenAI API KEY

3. `$ python3 main.py` This will run the python file.

### `PwnBERT.py`

From executing this file, you can acquire your training sets and eval sets, remember to modifiy     `generate_tokens()` function in `main` function.

### `train.py` and `train_v2.py`

Due to some problem we have not solve yet, we decided to create `train_v2.py` for trainning. 

Generally speaking: This is a Python script for fine-tuning the DistilBert model for sequence classification using PyTorch and the transformers library. The script defines a CodeDataset class that inherits from PyTorch's Dataset class, which represents a dataset of code files. The CodeDataset class loads code files from two directories, one containing vulnerable code and the other containing non-vulnerable code, and preprocesses the code using the DistilBertTokenizer to generate token IDs, attention masks, and labels.

The script then defines a compute_metrics function that calculates the accuracy of the model on the evaluation dataset. The main function of the script, finetune_pwnbert, loads the DistilBertForSequenceClassification model from the transformers library, initializes the model with a specified number of output labels, and fine-tunes the model on the training dataset. The function takes as input four directory paths containing the training and evaluation datasets of vulnerable and non-vulnerable code, respectively, and saves the finetuned model and tokenizer in the specified output directory.


## Update Logs
**Mainly updates after Mar 22, 2023:**

* v1.1, Mar 20: Started to use `concurrent.futures` for acceleration purposes.

* v1.2, Mar 22: Created `PwnBERT.py`, major adjust the structure of directories (because I need to import them), fix minor bugs and added new stuff on `generate_code_segments/` and `tokenize_codes`.

* v1.2.1: Fix bugs that might effect significantly on the codes