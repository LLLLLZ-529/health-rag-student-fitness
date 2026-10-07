#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_sft_1.5b.py — 体测健康领域指令微调（SFT + LoRA）1.5B版
内存/速度优化：动态padding（平均170 token）+ max_length=256 + batch=2
"""
import os, sys, json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "outputs_1.5b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 3
LR = 2e-4
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM = "你是一个大学生体测健康助手，擅长分析体测数据、预测健康趋势并给出运动建议。"


class InstructionDataset(Dataset):
    """不做固定padding，只截断；padding交给collator动态完成"""
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tok(text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            ids = b["input_ids"]
            pad_n = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_n)
            attn.append(b["attention_mask"] + [0] * pad_n)
            labels.append(ids + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    print("=" * 60)
    print("体测健康领域 SFT 微调 (1.5B, 动态padding)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    model_path = MODEL_NAME
    try:
        from modelscope import snapshot_download
        print(f"\n从 ModelScope 获取模型: {MODEL_NAME}")
        model_path = snapshot_download(MODEL_NAME, cache_dir=str(Path.home() / ".cache" / "modelscope"))
    except Exception as e:
        print(f"ModelScope失败({e})，使用HuggingFace")

    print(f"\n加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # 防swap
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    # 均衡采样5000条
    print("\n加载数据集...")
    random.seed(42)
    raw = []
    with open(DATA_DIR / "instruction_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    by_type = {}
    for item in raw:
        by_type.setdefault(item.get("task_type", "default"), []).append(item)
    per = 5000 // len(by_type)
    subset = []
    for t, items in by_type.items():
        random.shuffle(items)
        subset.extend(items[:per])
    random.shuffle(subset)
    subset_path = DATA_DIR / "instruction_dataset_train_5k.jsonl"
    with open(subset_path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_dataset = InstructionDataset(subset_path, tokenizer, MAX_LENGTH)
    val_full = InstructionDataset(DATA_DIR / "instruction_dataset_val.jsonl", tokenizer, MAX_LENGTH)
    val_dataset = torch.utils.data.Subset(val_full, range(min(100, len(val_full))))
    print(f"  训练集: {len(train_dataset)} 条（均衡采样）, 验证集: 100 条")

    collator = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.03), num_training_steps=total_steps)

    print(f"\n开始训练: {EPOCHS} epochs, {total_steps} steps")
    print(f"  batch={BATCH_SIZE}x{GRAD_ACCUM} (有效{BATCH_SIZE*GRAD_ACCUM}), maxlen={MAX_LENGTH}, lr={LR}")

    best_val_loss = float("inf")
    log_history = []
    import time
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_batches = 0, 0
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
                    elapsed = (time.time() - t_start) / 60
                    eta = elapsed / global_step * (total_steps - global_step)
                    print(f"  Ep{epoch+1}/{EPOCHS} Step {global_step}/{total_steps} "
                          f"loss={total_loss/n_batches:.4f} lr={scheduler.get_last_lr()[0]:.5f} "
                          f"[{elapsed:.0f}min, ETA {eta:.0f}min]")
                    log_history.append({"epoch": epoch+1, "step": global_step,
                                        "loss": total_loss/n_batches, "elapsed_min": elapsed})

        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE))
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_val = val_loss / val_batches
        print(f"\n  Epoch {epoch+1} 完成: train={total_loss/n_batches:.4f} val={avg_val:.4f}")
        if DEVICE == "mps":
            torch.mps.empty_cache()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = OUTPUT_DIR / "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  保存最佳模型: {save_path}")

    final_path = OUTPUT_DIR / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SFT完成! 最佳val_loss={best_val_loss:.4f}, 总耗时{(time.time()-t_start)/60:.0f}分钟")
    print("=" * 60)


if __name__ == "__main__":
    main()
