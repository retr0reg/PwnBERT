import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

import evaluate  # 导入你的评估库


class CodeDataset(Dataset):
    def __init__(self, vuln_dir, nvuln_dir, tokenizer):
        self.tokenizer = tokenizer
        self.vuln_files = os.listdir(vuln_dir)
        self.nvuln_files = os.listdir(nvuln_dir)
        self.vuln_dir = vuln_dir
        self.nvuln_dir = nvuln_dir

    def __len__(self):
        return len(self.vuln_files) + len(self.nvuln_files)

    def __getitem__(self, idx):
        if idx < len(self.vuln_files):
            file_path = os.path.join(self.vuln_dir, self.vuln_files[idx])
            label = 1
        else:
            file_path = os.path.join(self.nvuln_dir, self.nvuln_files[idx - len(self.vuln_files)])
            label = 0

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            inputs = self.tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].squeeze()
            attention_mask = inputs["attention_mask"].squeeze()
            # token_type_ids = inputs["token_type_ids"].squeeze()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "token_type_ids": token_type_ids,
                "labels": torch.tensor(label),
            }

def evaluate_pwnbert(vuln_eval_dir, nvuln_eval_dir, output_dir):
    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # 加载训练过的模型
    model = RobertaForSequenceClassification.from_pretrained(output_dir)
    
    device = torch.device("mps")
    model.to(device)
    model.eval()
    
    eval_dataset = CodeDataset(vuln_eval_dir, nvuln_eval_dir, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2)
    
    metric = evaluate.load("accuracy")
    total_eval_accuracy = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        total_eval_accuracy += metric.compute()["accuracy"]

    avg_eval_accuracy = total_eval_accuracy / len(eval_dataloader)
    print(f"Average evaluation accuracy: {avg_eval_accuracy:.2f}")


if __name__ == "__main__":
    vuln_eval_dir = "generate_code_segments/eval/vuln"
    nvuln_eval_dir = "generate_code_segments/eval/nvuln"
    output_dir = "pwnbert_finetuned"
    
    evaluate_pwnbert(vuln_eval_dir, nvuln_eval_dir, output_dir)
