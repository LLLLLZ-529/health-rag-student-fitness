#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_run_dpo_1.5b_v2.py — 严谨保守版DPO（修复v1崩溃问题）

修复点：
1. 只对response部分计算log prob（mask掉system+user prompt），sum over response tokens（TRL标准）
2. lr 5e-5 -> 5e-6（LoRA DPO需要极小lr，防止输出熵坍缩）
3. label smoothing ε=0.1，防止过度自信
4. 关闭gradient checkpointing（消除requires_grad warning，内存测试通过）
5. 1 epoch + 1000条，减少更新步数
6. 每50步greedy生成sanity check，检测重复退化
"""
import os, json, glob, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
BASE_DIR = Path(__file__).parent
SFT_MODEL = BASE_DIR / "outputs_1.5b" / "best_model"
OUTPUT_DIR = BASE_DIR / "outputs_1.5b" / "dpo_model_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_cands = glob.glob(str(Path.home() / ".cache/modelscope/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
BASE_MODEL_PATH = _cands[0] if _cands else "Qwen/Qwen2.5-1.5B-Instruct"

MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 1
LR = 5e-6          # v1是5e-5，降10倍
BETA = 0.1
LABEL_SMOOTH = 0.1
N_TRAIN = 1000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SYSTEM = "你是一个大学生体测健康助手，擅长分析体测数据并给出运动建议。"


class DPODataset(Dataset):
    """返回chosen/rejected的完整序列 + response mask（只assistant部分为1）"""
    def __init__(self, jsonl_path, tok, max_length=256):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok, self.max_length = tok, max_length

    def __len__(self):
        return len(self.data)

    def _encode(self, prompt, response):
        # prompt部分（到generation prompt为止）
        prompt_msgs = [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": prompt}]
        prompt_text = self.tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        # 完整序列
        full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]
        full_text = self.tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        full_ids = self.tok(full_text, truncation=True, max_length=self.max_length, add_special_tokens=False)["input_ids"]
        n_prompt = min(len(prompt_ids), len(full_ids))
        # response mask: prompt部分为0，response部分为1
        resp_mask = [0] * n_prompt + [1] * (len(full_ids) - n_prompt)
        return full_ids, [1] * len(full_ids), resp_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        c = self._encode(item["prompt"], item["chosen"])
        r = self._encode(item["prompt"], item["rejected"])
        return {"c": c, "r": r}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def _pad(self, triples):
        maxlen = max(len(t[0]) for t in triples)
        ids, attn, resp = [], [], []
        for seq_ids, seq_attn, seq_resp in triples:
            n = maxlen - len(seq_ids)
            ids.append(seq_ids + [self.pad_id] * n)
            attn.append(seq_attn + [0] * n)
            resp.append(seq_resp + [0] * n)
        return (torch.tensor(ids), torch.tensor(attn), torch.tensor(resp))

    def __call__(self, batch):
        c = self._pad([b["c"] for b in batch])
        r = self._pad([b["r"] for b in batch])
        return {"c": c, "r": r}


def response_log_prob(model, ids, attn, resp_mask):
    """只对response tokens求sum log prob（标准DPO）"""
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    labels = ids[:, 1:]
    # 对齐：预测token t的logit在位置t-1，所以resp_mask取[1:]
    m = resp_mask[:, 1:].float()
    logp = F.log_softmax(logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return (logp * m).sum(-1)  # sum over response tokens，不做长度归一化


def sanity_generate(model, tok):
    """生成一句推荐，检测是否退化（重复单字）"""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "该学生最弱的指标是耐力（800米跑不及格），请给出运动建议。"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    # 退化检测：最高频单字占比
    if len(gen) > 10:
        top_ratio = max(gen.count(c) for c in set(gen)) / len(gen)
    else:
        top_ratio = 1.0
    return gen, top_ratio


def main():
    print("=" * 64)
    print("DPO v2（保守版：response-only sum logprob, lr=5e-6, label_smooth）")
    print("=" * 64)
    tok = AutoTokenizer.from_pretrained(str(SFT_MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"加载模型: {BASE_MODEL_PATH}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(SFT_MODEL))
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)
    model.config.use_cache = False
    model.enable_input_require_grads()
    # 故意不启用gradient checkpointing
    model.to(DEVICE)
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n加载数据...")
    full = DPODataset(BASE_DIR / "dpo_dataset.jsonl", tok, MAX_LENGTH)
    random.seed(42)
    idx = list(range(len(full))); random.shuffle(idx)
    train_ds = torch.utils.data.Subset(full, idx[:N_TRAIN])
    val_full = DPODataset(BASE_DIR / "dpo_dataset_val.jsonl", tok, MAX_LENGTH)
    val_ds = torch.utils.data.Subset(val_full, range(min(50, len(val_full))))
    print(f"  训练: {len(train_ds)}, 验证: {len(val_ds)}")

    coll = Collator(tok.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=coll)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    print(f"  {EPOCHS}ep, {total_steps}steps, beta={BETA}, lr={LR}, smooth={LABEL_SMOOTH}")

    # 训练前baseline生成
    model.eval()
    gen0, r0 = sanity_generate(model, tok)
    print(f"\n[训练前] 重复率={r0:.2f} 生成: {gen0[:80]}")

    best = float("inf"); t0 = time.time(); step_count = 0
    for epoch in range(EPOCHS):
        model.train(); tot, n = 0.0, 0; opt.zero_grad()
        for step, b in enumerate(train_loader):
            c_ids, c_attn, c_resp = [x.to(DEVICE) for x in b["c"]]
            r_ids, r_attn, r_resp = [x.to(DEVICE) for x in b["r"]]
            model.train()
            pc = response_log_prob(model, c_ids, c_attn, c_resp)
            pr = response_log_prob(model, r_ids, r_attn, r_resp)
            with torch.no_grad(), model.disable_adapter():
                rc = response_log_prob(model, c_ids, c_attn, c_resp)
                rr = response_log_prob(model, r_ids, r_attn, r_resp)
            logits = BETA * ((pc - rc) - (pr - rr))
            # label-smoothed DPO loss
            loss = (-(1 - LABEL_SMOOTH) * F.logsigmoid(logits)
                    - LABEL_SMOOTH * F.logsigmoid(-logits)).mean() / GRAD_ACCUM
            loss.backward()
            tot += loss.item() * GRAD_ACCUM; n += 1
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(); step_count += 1
                if step_count % 10 == 0:
                    el = (time.time() - t0) / 60
                    print(f"  Step {step_count}/{total_steps} loss={tot/n:.4f} [{el:.0f}min]")
                # 每50步sanity check
                if step_count % 50 == 0:
                    model.eval()
                    gen, ratio = sanity_generate(model, tok)
                    flag = " ⚠️退化!" if ratio > 0.5 else " ok"
                    print(f"  [check {step_count}] 重复率={ratio:.2f}{flag}: {gen[:60]}")
                    if ratio > 0.5:
                        print("  检测到退化，提前终止训练！")
                        torch.save({"early_stop": True, "step": step_count}, OUTPUT_DIR / "DEGENERATED.flag")
                        return
                    model.train()

        # 验证
        model.eval(); vl, vn = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                c_ids, c_attn, c_resp = [x.to(DEVICE) for x in b["c"]]
                r_ids, r_attn, r_resp = [x.to(DEVICE) for x in b["r"]]
                pc = response_log_prob(model, c_ids, c_attn, c_resp)
                pr = response_log_prob(model, r_ids, r_attn, r_resp)
                with model.disable_adapter():
                    rc = response_log_prob(model, c_ids, c_attn, c_resp)
                    rr = response_log_prob(model, r_ids, r_attn, r_resp)
                logits = BETA * ((pc - rc) - (pr - rr))
                vl += (-(1 - LABEL_SMOOTH) * F.logsigmoid(logits)
                       - LABEL_SMOOTH * F.logsigmoid(-logits)).mean().item(); vn += 1
        avg_vl, avg_tl = vl / vn, tot / n
        print(f"\n  Epoch完成: train={avg_tl:.4f} val={avg_vl:.4f}")
        if DEVICE == "mps": torch.mps.empty_cache()
        if avg_vl < best:
            best = avg_vl
            model.save_pretrained(str(OUTPUT_DIR)); tok.save_pretrained(str(OUTPUT_DIR))
            print(f"  保存DPO v2模型: {OUTPUT_DIR}")

    # 最终生成确认
    model.eval()
    gen_f, r_f = sanity_generate(model, tok)
    print(f"\n[训练后] 重复率={r_f:.2f} 生成: {gen_f[:100]}")
    print("\n" + "=" * 64)
    print(f"DPO v2完成! best_val={best:.4f}, 耗时{(time.time()-t0)/60:.0f}分钟")
    print("=" * 64)


if __name__ == "__main__":
    main()
