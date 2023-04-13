import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
# from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification


from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AdamW
from transformers import AutoConfig

from transformers.optimization import get_linear_schedule_with_warmup
from torch import nn
from torch.optim import AdamW

from typing import Optional



# class CustomTrainer(Trainer):
#     def create_optimizer(self) -> torch.optim.Optimizer:
#         optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
#         self.optimizer = optimizer
#         return optimizer

#     def create_scheduler(self, num_training_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
#         scheduler = get_linear_schedule_with_warmup(
#             self.optimizer,
#             num_warmup_steps=0,
#             num_training_steps=num_training_steps,
#         )
#         self.lr_scheduler = scheduler
#         return scheduler




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
        
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def finetune_pwnbert(vuln_dir, nvuln_dir, vuln_eval_dir, nvuln_eval_dir, model_name = "microsoft/codebert-base", output_dir="./pwnbert_finetuned"):
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    
    #### add accelerators ###
    from accelerate import Accelerator
    from tqdm import tqdm
    from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
    from torch.utils.tensorboard import SummaryWriter
    
    writer = SummaryWriter("logs")
    accelerator = Accelerator()
    
    
    
    # device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    device = torch.device("mps")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    
    
    train_dataset = CodeDataset(vuln_dir, nvuln_dir, tokenizer)
    # eval_dataset = CodeDataset(vuln_eval_dir, nvuln_eval_dir, tokenizer)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=2)

    
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer
    )
    
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            print(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step = epoch * len(train_dataloader) + step
            writer.add_scalar("Loss/train", loss.item(), global_step)
            
            progress_bar.update(1)
            
    writer.close()
    try:
        model.save_pretrained("pwnbert_finetuned")
        tokenizer.save_pretrained("pwnbert_finetuned")
        print("Model and tokenizer saved successfully.")
    except Exception as e:
        print("Error occurred while saving the model and tokenizer:", e)
        
def train_s1():
    vuln_dir = "generate_code_segments/vuln"
    nvuln_dir = "generate_code_segments/nvuln"
    vuln_eval_dir = "generate_code_segments/eval/vuln"
    nvuln_eval_dir = "generate_code_segments/eval/nvuln"
    finetune_pwnbert(vuln_dir, nvuln_dir, vuln_eval_dir, nvuln_eval_dir)
    
def train_s2():
    vuln_dir = "outputs/vuln"
    nvuln_dir = "outputs/nvuln"
    vuln_eval_dir = ""
    nvuln_eval_dir = ""
    finetune_pwnbert(vuln_dir, nvuln_dir, vuln_eval_dir, nvuln_eval_dir)

if __name__ == "__main__":
    train_s2()