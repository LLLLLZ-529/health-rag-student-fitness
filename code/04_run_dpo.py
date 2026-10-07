#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_run_dpo.py — DPO偏好对齐训练

基于SFT模型，用DPO损失对齐推荐偏好。
DPO loss: -E[log σ(β * (π_chosen/π_ref_chosen - π_rejected/π_ref_rejected))]
"""
import os, sys, json, glob
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE_DIR = Path(__file__).parent
SFT_MODEL = BASE_DIR / "outputs" / "best_model"
_cands = sorted(glob.glob(str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/*")))
BASE_MODEL_PATH = _cands[-1] if _cands else ""
assert BASE_MODEL_PATH, "未找到 Qwen2.5-0.5B 本地模型，请先从 ModelScope 下载或设置模型路径"
OUTPUT_DIR = BASE_DIR / "outputs" / "dpo_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 2
LR = 5e-5
BETA = 0.1  # DPO温度系数

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class DPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _encode(self, prompt, response):
        messages = [
            {"role": "system", "content": "你是一个大学生体测健康助手，擅长分析体测数据并给出运动建议。"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze()

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_ids, chosen_mask = self._encode(item["prompt"], item["chosen"])
        rejected_ids, rejected_mask = self._encode(item["prompt"], item["rejected"])
        return {
            "chosen_ids": chosen_ids, "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids, "rejected_mask": rejected_mask,
        }


def get_log_probs(model, input_ids, attention_mask):
    """计算序列的log prob（只算非padding部分）"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def main():
    print("=" * 60)
    print("DPO 偏好对齐训练")
    print("=" * 60)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MODEL), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载策略模型（SFT + LoRA，可训练）
    print(f"\n加载策略模型: {SFT_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    policy_model = PeftModel.from_pretrained(base_model, str(SFT_MODEL))
    # 确保只有LoRA参数可训练
    for name, param in policy_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    policy_model.config.use_cache = False
    policy_model.enable_input_require_grads()
    policy_model.gradient_checkpointing_enable()
    policy_model.to(DEVICE)
    n_trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy_model.parameters())
    print(f"  可训练参数: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    # 加载参考模型（SFT，冻结）
    print("加载参考模型（冻结）...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    ref_model = PeftModel.from_pretrained(ref_base, str(SFT_MODEL))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(DEVICE)

    # 加载数据
    print("\n加载DPO数据集...")
    train_dataset = DPODataset(BASE_DIR / "dpo_dataset.jsonl", tokenizer, MAX_LENGTH)
    val_dataset = DPODataset(BASE_DIR / "dpo_dataset_val.jsonl", tokenizer, MAX_LENGTH)
    print(f"  训练集: {len(train_dataset)} 条")
    print(f"  验证集: {len(val_dataset)} 条")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 优化器（只训练LoRA参数）
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM

    print(f"\n开始DPO训练: {EPOCHS} epochs, {total_steps} steps")
    print(f"  beta={BETA}, lr={LR}, batch={BATCH_SIZE}x{GRAD_ACCUM}")

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        policy_model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)

            # 策略模型log probs
            pi_chosen = get_log_probs(policy_model, chosen_ids, chosen_mask)
            pi_rejected = get_log_probs(policy_model, rejected_ids, rejected_mask)

            # 参考模型log probs（无梯度）
            with torch.no_grad():
                ref_chosen = get_log_probs(ref_model, chosen_ids, chosen_mask)
                ref_rejected = get_log_probs(ref_model, rejected_ids, rejected_mask)

            # DPO loss
            logits = BETA * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
            loss = -F.logsigmoid(logits).mean() / GRAD_ACCUM
            loss.backward()

            total_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step = (step + 1) // GRAD_ACCUM + epoch * (len(train_loader) // GRAD_ACCUM)
                if global_step % 10 == 0:
                    avg_loss = total_loss / n_batches
                    print(f"  Epoch {epoch+1}/{EPOCHS} Step {global_step}/{total_steps} loss={avg_loss:.4f}")

        # 验证
        policy_model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                chosen_ids = batch["chosen_ids"].to(DEVICE)
                chosen_mask = batch["chosen_mask"].to(DEVICE)
                rejected_ids = batch["rejected_ids"].to(DEVICE)
                rejected_mask = batch["rejected_mask"].to(DEVICE)
                pi_chosen = get_log_probs(policy_model, chosen_ids, chosen_mask)
                pi_rejected = get_log_probs(policy_model, rejected_ids, rejected_mask)
                ref_chosen = get_log_probs(ref_model, chosen_ids, chosen_mask)
                ref_rejected = get_log_probs(ref_model, rejected_ids, rejected_mask)
                logits = BETA * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
                loss = -F.logsigmoid(logits).mean()
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_train_loss = total_loss / n_batches
        print(f"\n  Epoch {epoch+1} 完成: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            policy_model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print(f"  保存最佳DPO模型到: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("DPO训练完成!")
    print(f"  最佳验证loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
