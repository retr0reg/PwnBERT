import os
import re
import sys
from pathlib import Path
from pycparser import c_parser
import concurrent.futures
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed

openai.api_key = "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"

def remove_comments(text):
    text = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", text)  # Remove /* ... */ comments
    text = re.sub(re.compile("//.*?\n"), "", text)  # Remove // comments
    return text

def rename_functions(text):
    func_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    func_counter = 1
    func_dict = {}

    def replace_func_name(match):
        nonlocal func_counter
        name = match.group(1)
        if name not in func_dict:
            func_dict[name] = f"func_{func_counter}"
            func_counter += 1
        return f"{func_dict[name]}("

    return func_pattern.sub(replace_func_name, text)

def rename_variables(text):
    var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=')
    var_counter = 1
    var_dict = {}

    def replace_var_name(match):
        nonlocal var_counter
        name = match.group(1)
        if name not in var_dict:
            var_dict[name] = f"var_{var_counter}"
            var_counter += 1
        return f"{var_dict[name]} ="

    return var_pattern.sub(replace_var_name, text)

def process_file(file_path, output_path,choice):
    with open(file_path, 'r') as f:
        content = f.read()

    content = remove_comments(content)
    content = rename_functions(content)
    content = rename_variables(content)
    
    if choice:
        content = process_nvuln(content)
    else:
        pass

    with open(output_path, 'w') as f:
        f.write(content)
        
    return content

def process_directory(input_dir, output_dir, nvuln=False):
    num = []
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files_to_process = list(input_path.glob('*.c'))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, output_path / file.name, int(nvuln)): file for file in files_to_process}
        
        with tqdm(total=len(files_to_process), desc="Processing files") as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"{file}: {e}")
                progress_bar.update(1)

    return len(files_to_process)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def chat_api(payload):
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


def process_nvuln(content):
    prompt = f"I have a vulnerable C code snippet, and I'd like you to generate a similar code snippet that is secure and free of vulnerabilities. Please provide only the code without any comments. Here's the vulnerable code:{content}"
    result = chat_api(prompt)["choices"][0]["message"]["content"]
    result = "#include" + result.split("#include")[-1]
    return result

if __name__ == "__main__":
    vuln_amount = 0
    nvuln_amount = 0
    input_dir = ""
    for i in range(1,2+1):
        vuln_amount += process_directory(input_dir=f"juliet-test-suite-c/testcases/CWE121_Stack_Based_Buffer_Overflow/s0{i}", output_dir="outputs/vuln")
        nvuln_amount += process_directory(input_dir=f"juliet-test-suite-c/testcases/CWE121_Stack_Based_Buffer_Overflow/s0{i}", output_dir="outputs/nvuln", nvuln=True)[0]
        
    print(f"Total {vuln_amount} vuln files were processed, {nvuln_amount} non-vuln file were processed; Totally {vuln_amount+nvuln_amount} files were processed")