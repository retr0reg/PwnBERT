import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AdamW

from transformers.optimization import get_linear_schedule_with_warmup
from torch import nn


class CustomTrainer(Trainer):
    def create_optimizer(self) -> torch.optim.Optimizer:
        # Replace this with the optimizer of your choice
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer = optimizer  # Add this line
        return optimizer


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


def finetune_pwnbert(vuln_dir, nvuln_dir, vuln_eval_dir, nvuln_eval_dir, model_name="bert-base-cased", output_dir="./pwnbert_finetuned"):
    
    tokenizer = BertTokenizer.from_pretrained(model_name)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    train_dataset = CodeDataset(vuln_dir, nvuln_dir, tokenizer)
    eval_dataset = CodeDataset(vuln_eval_dir, nvuln_eval_dir, tokenizer)

    EPOCHS = 30
    

    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=EPOCHS,
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_dir="logs",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    # 使用DataCollatorWithPadding来处理批次的填充
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    
    #### NEW APPROACH ####
    
    EPOCHS = 20
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_dataset) * EPOCHS // training_args.per_device_train_batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(training_args.device)

    #### END NEW APPROACH ####



    # 创建训练器
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],  # Add the callback here
    )


    trainer.train()

    return model, tokenizer

if __name__ == "__main__":
    vuln_dir = "generate_code_segments/vuln"
    nvuln_dir = "generate_code_segments/nvuln"
    vuln_eval_dir = "generate_code_segments/eval/vuln"
    nvuln_eval_dir = "generate_code_segments/eval/nvuln"
    model, tokenizer = finetune_pwnbert(vuln_dir, nvuln_dir, vuln_eval_dir, nvuln_eval_dir)
    model.save_pretrained("./pwnbert_finetuned")
    tokenizer.save_pretrained("./pwnbert_finetuned")