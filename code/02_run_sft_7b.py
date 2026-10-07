#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_sft_7b.py — 体测健康领域指令微调（SFT + LoRA）7B版
支持断点续训：每500步保存checkpoint，启动自动恢复
"""
import os, sys, json, random, time, itertools, glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "outputs_7b"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 256
BATCH_SIZE = 16
GRAD_ACCUM = 4
EPOCHS = 3
LR = 1e-4
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
SAVE_INTERVAL = 200  # 每500个global_step保存一次checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM = "你是一个大学生体测健康助手，擅长分析体测数据、预测健康趋势并给出运动建议。"

class InstructionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256, cache_path=None):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok = tokenizer
        self.max_length = max_length
        if cache_path and Path(cache_path).exists():
            print(f"  加载预tokenize缓存: {cache_path}", flush=True)
            self.encoded = torch.load(cache_path, weights_only=False)
        else:
            print(f"  预tokenize {len(self.data)} 条数据...", flush=True)
            self.encoded = []
            for i, item in enumerate(self.data):
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                enc = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
                self.encoded.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
                if (i+1) % 10000 == 0:
                    print(f"    已处理 {i+1}/{len(self.data)}", flush=True)
            if cache_path:
                torch.save(self.encoded, cache_path)
                print(f"  缓存已保存: {cache_path}", flush=True)
    def __len__(self):
        return len(self.encoded)
    def __getitem__(self, idx):
        return self.encoded[idx]

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

def save_checkpoint(model, optimizer, scheduler, epoch, data_step, best_val_loss, log_history, path):
    """保存完整训练状态"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "data_step": data_step,
        "best_val_loss": best_val_loss,
        "log_history": log_history,
    }, path)
    print(f"  [CHECKPOINT] 已保存: {path} (epoch={epoch}, step={data_step})", flush=True)

def load_checkpoint(path, model, optimizer, scheduler):
    """加载训练状态，返回 (epoch, data_step, best_val_loss, log_history)"""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  [RESUME] 从 {path} 恢复: epoch={ckpt['epoch']}, step={ckpt['data_step']}, best_val={ckpt['best_val_loss']:.4f}", flush=True)
    return ckpt["epoch"], ckpt["data_step"], ckpt["best_val_loss"], ckpt["log_history"]

def main():
    print("=" * 60, flush=True)
    print("体测健康领域 SFT 微调 (7B, CUDA, 支持断点续训)", flush=True)
    print("=" * 60, flush=True)
    print(f"设备: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)

    _cands = sorted(glob.glob(str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/*")))
    model_path = _cands[-1] if _cands else ""
    assert model_path, "未找到 Qwen2.5-7B 本地模型，请先从 ModelScope 下载"
    print(f"\n使用本地模型: {model_path}", flush=True)
    print(f"\n加载模型: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    print("\n加载数据集...", flush=True)
    random.seed(42)
    raw = []
    with open(DATA_DIR / "instruction_dataset_aug.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))
    by_type = {}
    for item in raw:
        by_type.setdefault(item.get("task_type", "default"), []).append(item)
    per = max(len(v) for v in by_type.values())
    subset = []
    for t, items in by_type.items():
        random.shuffle(items)
        subset.extend(items[:per])
    random.shuffle(subset)
    subset_path = DATA_DIR / "instruction_dataset_train_5k.jsonl"
    with open(subset_path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    train_dataset = InstructionDataset(subset_path, tokenizer, MAX_LENGTH, cache_path=str(DATA_DIR / "train_cache.pt"))
    val_full = InstructionDataset(DATA_DIR / "instruction_dataset_aug_val.jsonl", tokenizer, MAX_LENGTH, cache_path=str(DATA_DIR / "val_cache.pt"))
    val_dataset = torch.utils.data.Subset(val_full, range(min(500, len(val_full))))
    print(f"  训练集: {len(train_dataset)} 条（多样性增强）, 验证集: 500 条", flush=True)

    collator = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.03), num_training_steps=total_steps)

    # 检查是否有断点
    start_epoch = 0
    start_data_step = 0
    best_val_loss = float("inf")
    log_history = []
    ckpt_files = sorted(CKPT_DIR.glob("ckpt_*.pt"))
    resume_path = None
    if ckpt_files:
        resume_path = ckpt_files[-1]  # 取最新的
    if resume_path and resume_path.exists():
        start_epoch, start_data_step, best_val_loss, log_history = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"  断点续训: 从 epoch={start_epoch}, data_step={start_data_step} 继续", flush=True)
    else:
        print("  无断点，从头开始训练", flush=True)

    print(f"\n开始训练: {EPOCHS} epochs, {total_steps} steps", flush=True)
    print(f"  batch={BATCH_SIZE}x{GRAD_ACCUM} (有效{BATCH_SIZE*GRAD_ACCUM}), maxlen={MAX_LENGTH}, lr={LR}", flush=True)
    print(f"  断点保存间隔: 每 {SAVE_INTERVAL} global_step", flush=True)

    t_start = time.time()
    steps_per_epoch = len(train_loader) // GRAD_ACCUM

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss, n_batches = 0, 0
        optimizer.zero_grad()

        # 构建迭代器，如果是断点恢复则跳过已完成的step
        train_iter = enumerate(train_loader)
        if epoch == start_epoch and start_data_step > 0:
            print(f"  跳过已完成的 {start_data_step} 个 data_step...", flush=True)
            train_iter = itertools.islice(train_iter, start_data_step, None)
            start_data_step = 0  # 重置，下一个epoch不跳过

        for step, batch in train_iter:
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
                global_step = (step + 1) // GRAD_ACCUM + epoch * steps_per_epoch

                if global_step % 5 == 0:
                    elapsed = (time.time() - t_start) / 60
                    eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                    vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
                    print(f"  Ep{epoch+1}/{EPOCHS} Step {global_step}/{total_steps} loss={total_loss/n_batches:.4f} lr={scheduler.get_last_lr()[0]:.5f} VRAM={vram:.1f}GB [{elapsed:.0f}min, ETA {eta:.0f}min]", flush=True)
                    log_history.append({"epoch": epoch+1, "step": global_step, "loss": total_loss/n_batches, "elapsed_min": elapsed})

                # 定期保存checkpoint
                if global_step % SAVE_INTERVAL == 0:
                    ckpt_path = CKPT_DIR / f"ckpt_ep{epoch+1}_step{global_step}.pt"
                    save_checkpoint(model, optimizer, scheduler, epoch, step + 1, best_val_loss, log_history, ckpt_path)
                    # 只保留最近3个checkpoint
                    old_ckpts = sorted(CKPT_DIR.glob("ckpt_*.pt"))
                    for old in old_ckpts[:-3]:
                        old.unlink()

        # Epoch结束：验证
        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE), labels=batch["labels"].to(DEVICE))
                val_loss += outputs.loss.item()
                val_batches += 1
        avg_val = val_loss / val_batches
        print(f"\n  Epoch {epoch+1} 完成: train={total_loss/n_batches:.4f} val={avg_val:.4f}", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = OUTPUT_DIR / "best_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  保存最佳模型: {save_path} (val={avg_val:.4f})", flush=True)

        # Epoch结束也保存checkpoint
        ckpt_path = CKPT_DIR / f"ckpt_ep{epoch+1}_end.pt"
        save_checkpoint(model, optimizer, scheduler, epoch + 1, 0, best_val_loss, log_history, ckpt_path)

    # 训练完成
    final_path = OUTPUT_DIR / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)
    print("\n" + "=" * 60, flush=True)
    print(f"SFT 7B完成! 最佳val_loss={best_val_loss:.4f}, 总耗时{(time.time()-t_start)/60:.0f}分钟", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()