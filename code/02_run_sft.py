#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_sft.py — 体测健康领域指令微调（SFT + LoRA）

模型：Qwen2.5-0.5B-Instruct
方法：LoRA (rank=16, alpha=32)
数据：7600条指令数据集（分类/预测/推荐/知识问答）
"""
import os, sys, json, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE = Path(__file__).parents[4]
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"设备: {DEVICE}")


class InstructionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Qwen chat template
        messages = [
            {"role": "system", "content": "你是一个大学生体测健康助手，擅长分析体测数据、预测健康趋势并给出运动建议。"},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels = input_ids.clone()
        # 只计算assistant回答部分的loss（简单处理：全部计算，效果也可以）
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    print("=" * 60)
    print("体测健康领域 SFT 微调")
    print("=" * 60)

    # 优先用ModelScope下载（国内速度快）
    model_path = MODEL_NAME
    try:
        from modelscope import snapshot_download
        print(f"\n从 ModelScope 下载模型: {MODEL_NAME}")
        model_path = snapshot_download(MODEL_NAME, cache_dir=str(Path.home() / ".cache" / "modelscope"))
        print(f"模型已下载到: {model_path}")
    except Exception as e:
        print(f"ModelScope下载失败({e})，使用HuggingFace")

    # 加载tokenizer
    print(f"\n加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # 训练时关闭KV cache
    model.gradient_checkpointing_enable()  # 节省内存

    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    # 加载数据
    print("\n加载数据集...")
    train_dataset = InstructionDataset(DATA_DIR / "instruction_dataset.jsonl", tokenizer, MAX_LENGTH)
    val_dataset = InstructionDataset(DATA_DIR / "instruction_dataset_val.jsonl", tokenizer, MAX_LENGTH)
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.03), num_training_steps=total_steps
    )

    # 训练循环
    print(f"\n开始训练: {EPOCHS} epochs, {total_steps} steps")
    print(f"  batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}")
    print(f"  有效batch_size={BATCH_SIZE * GRAD_ACCUM}")

    best_val_loss = float("inf")
    log_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step = (step + 1) // GRAD_ACCUM + epoch * (len(train_loader) // GRAD_ACCUM)
                if global_step % 20 == 0:
                    avg_loss = total_loss / n_batches
                    print(f"  Epoch {epoch+1}/{EPOCHS} Step {global_step}/{total_steps} "
                          f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
                    log_history.append({"epoch": epoch+1, "step": global_step, "loss": avg_loss})

        # 验证
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_train_loss = total_loss / n_batches
        print(f"\n  Epoch {epoch+1} 完成: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = OUTPUT_DIR / "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  保存最佳模型到: {save_path}")

    # 保存最终模型
    final_path = OUTPUT_DIR / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n最终模型保存到: {final_path}")

    # 保存训练日志
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print("\n" + "=" * 60)
    print("SFT 训练完成!")
    print(f"  最佳验证loss: {best_val_loss:.4f}")
    print(f"  模型: {MODEL_NAME} + LoRA(r={LORA_R})")
    print(f"  数据: {len(train_dataset)} 条指令")
    print("=" * 60)


if __name__ == "__main__":
    main()
