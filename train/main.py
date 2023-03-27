import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def get_file_location(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

class CodeDataset(Dataset):
    def __init__(self, vuln_path, nvuln_path, tokenizer, mode='train', split_ratio=0.8):
        self.tokenizer = tokenizer
        self.mode = mode
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

        samples_train, samples_eval, labels_train, labels_eval = train_test_split(self.samples,self.labels, train_size=split_ratio, random_state=42)
        if mode == 'train':
            self.samples = samples_train
            self.labels = labels_train
        else:
            self.samples = samples_eval
            self.labels = labels_eval

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code = self.samples[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        tokenized["labels"] = torch.tensor(label, dtype=torch.long)
        return {key: torch.squeeze(val, 0) for key, val in tokenized.items()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean()}

def main():
    model_name = "microsoft/codebert-base"
    # model_name = "EleutherAI/gpt-neo-1.3B"
    vuln_path = get_file_location("../generate_code_segments/vuln")
    nvuln_path = get_file_location("../generate_code_segments/nvuln")
    epochs = 3

        # 初始化 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 创建数据集
    train_dataset = CodeDataset(vuln_path, nvuln_path, tokenizer, mode='train')
    eval_dataset = CodeDataset(vuln_path, nvuln_path, tokenizer, mode='eval')

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none"
        )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )

    # 开始微调
    trainer.train()



if __name__ == "__main__":
    main()
