#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_run_dpo_1.5b.py — DPO偏好对齐（1.5B，单模型+动态padding省内存）
参考模型通过 disable_adapter() 切换，零额外模型内存。
"""
import os, json, glob, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE_DIR = Path(__file__).parent
SFT_MODEL = BASE_DIR / "outputs_1.5b" / "best_model"
OUTPUT_DIR = BASE_DIR / "outputs_1.5b" / "dpo_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_cands = glob.glob(str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
BASE_MODEL_PATH = _cands[0] if _cands else "Qwen/Qwen2.5-1.5B-Instruct"

MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 2
LR = 5e-5
BETA = 0.1
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SYSTEM = "你是一个大学生体测健康助手，擅长分析体测数据并给出运动建议。"


class DPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _enc(self, prompt, response):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return self.tok(text, truncation=True, max_length=self.max_length, add_special_tokens=False)

    def __getitem__(self, idx):
        item = self.data[idx]
        c = self._enc(item["prompt"], item["chosen"])
        r = self._enc(item["prompt"], item["rejected"])
        return {"c_ids": c["input_ids"], "c_mask": c["attention_mask"],
                "r_ids": r["input_ids"], "r_mask": r["attention_mask"]}


class DPOCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def _pad(self, seqs):
        maxlen = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            n = maxlen - len(s)
            ids.append(s + [self.pad_id] * n)
            mask.append([1] * len(s) + [0] * n)
        return torch.tensor(ids), torch.tensor(mask)

    def __call__(self, batch):
        c_ids, c_mask = self._pad([b["c_ids"] for b in batch])
        r_ids, r_mask = self._pad([b["r_ids"] for b in batch])
        return {"c_ids": c_ids, "c_mask": c_mask, "r_ids": r_ids, "r_mask": r_mask}


def mean_log_prob(model, ids, mask):
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1, :]
    labels = ids[:, 1:]
    m = mask[:, 1:].float()
    lp = F.log_softmax(logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return (lp * m).sum(-1) / m.sum(-1).clamp(min=1)


def main():
    import time
    print("=" * 60)
    print("DPO 偏好对齐 (1.5B, 单模型+动态padding)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MODEL), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载模型: {BASE_MODEL_PATH}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(SFT_MODEL))
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()  # 防swap
    model.to(DEVICE)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_tr:,}; 参考模型=disable_adapter()切换")

    print("\n加载DPO数据集...")
    full = DPODataset(BASE_DIR / "dpo_dataset.jsonl", tokenizer, MAX_LENGTH)
    random.seed(42)
    idx = list(range(len(full))); random.shuffle(idx)
    train_ds = torch.utils.data.Subset(full, idx[:2000])
    val_full = DPODataset(BASE_DIR / "dpo_dataset_val.jsonl", tokenizer, MAX_LENGTH)
    val_ds = torch.utils.data.Subset(val_full, range(min(50, len(val_full))))
    print(f"  训练: {len(train_ds)} 条, 验证: {len(val_ds)} 条")

    collator = DPOCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    print(f"\n开始: {EPOCHS}ep, {total_steps}steps, beta={BETA}, lr={LR}")

    best = float("inf")
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        optimizer.zero_grad()
        for step, b in enumerate(train_loader):
            c_ids, c_mask = b["c_ids"].to(DEVICE), b["c_mask"].to(DEVICE)
            r_ids, r_mask = b["r_ids"].to(DEVICE), b["r_mask"].to(DEVICE)

            model.train()
            pi_c = mean_log_prob(model, c_ids, c_mask)
            pi_r = mean_log_prob(model, r_ids, r_mask)
            with torch.no_grad(), model.disable_adapter():
                ref_c = mean_log_prob(model, c_ids, c_mask)
                ref_r = mean_log_prob(model, r_ids, r_mask)

            logits = BETA * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(logits).mean() / GRAD_ACCUM
            loss.backward()
            tot += loss.item() * GRAD_ACCUM
            n += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); optimizer.zero_grad()
                gs = (step + 1) // GRAD_ACCUM + epoch * (len(train_loader) // GRAD_ACCUM)
                if gs % 10 == 0:
                    el = (time.time() - t0) / 60
                    print(f"  Ep{epoch+1}/{EPOCHS} Step {gs}/{total_steps} loss={tot/n:.4f} [{el:.0f}min]")

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                c_ids, c_mask = b["c_ids"].to(DEVICE), b["c_mask"].to(DEVICE)
                r_ids, r_mask = b["r_ids"].to(DEVICE), b["r_mask"].to(DEVICE)
                pi_c = mean_log_prob(model, c_ids, c_mask)
                pi_r = mean_log_prob(model, r_ids, r_mask)
                with model.disable_adapter():
                    ref_c = mean_log_prob(model, c_ids, c_mask)
                    ref_r = mean_log_prob(model, r_ids, r_mask)
                logits = BETA * ((pi_c - ref_c) - (pi_r - ref_r))
                vl += -F.logsigmoid(logits).mean().item(); vn += 1
        avg_vl, avg_tl = vl / vn, tot / n
        print(f"\n  Epoch {epoch+1} 完成: train={avg_tl:.4f} val={avg_vl:.4f}")
        if DEVICE == "mps":
            torch.mps.empty_cache()
        if avg_vl < best:
            best = avg_vl
            model.save_pretrained(str(OUTPUT_DIR))
            tokenizer.save_pretrained(str(OUTPUT_DIR))
            print(f"  保存最佳DPO模型: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print(f"DPO完成! best_val={best:.4f}, 耗时{(time.time()-t0)/60:.0f}分钟")
    print("=" * 60)


if __name__ == "__main__":
    main()
