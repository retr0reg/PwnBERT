import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

class CodeDataset(Dataset):
    def __init__(self, vuln_path, nvuln_path, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        self.labels = []

        for file in os.listdir(vuln_path):
            with open(os.path.join(vuln_path, file), 'r') as f:
                code = f.read()
                self.samples.append(code)
                self.labels.append(1)  # 有漏洞的代码标签为1

        for file in os.listdir(nvuln_path):
            with open(os.path.join(nvuln_path, file), 'r') as f:
                code = f.read()
                self.samples.append(code)
                self.labels.append(0)  # 没有漏洞的代码标签为0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code = self.samples[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = torch.tensor(label, dtype=torch.long)
        return tokenized

def main():
    model_name = "microsoft/codebert-base"
    vuln_path = "vuln/"
    nvuln_path = "nvuln/"
    epochs = 3

    # 初始化 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 创建数据集
    dataset = CodeDataset(vuln_path, nvuln_path, tokenizer)

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # 定义评价指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == labels).mean()}

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )

    # 开始微调
    trainer.train()

if __name__ == "__main__":
    main()
