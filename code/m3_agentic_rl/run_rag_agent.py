#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_rag_agent.py — RL Agent + RAG 知识库对比实验"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from environment.simulator import HealthInterventionEnv, sample_student_from_data, ACTIONS, INDICATORS, WEIGHTS, StudentState
from agents.base import RuleAgent
from agents.rl_agent import RLAgent
from training.trainer import evaluate_agent
from utils.rag_retriever import RAGRetriever, generate_recommendation, ACTION_NAMES_CN

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR = BASE_DIR.parents[3] / "chunks" / "chunks"
DEVICE = "mps" if __import__("torch").backends.mps.is_available() else "cpu"


class RAGWrapperAgent:
    def __init__(self, base_agent, retriever):
        self.base_agent = base_agent
        self.retriever = retriever
        self.name = f"{base_agent.name}+RAG"
        self.last_recommendation = None
        self.last_retrieved = []

    def select_action(self, state, context=None):
        action = self.base_agent.select_action(state, context)
        self.last_retrieved = self.retriever.retrieve_for_action(action, state)
        self.last_recommendation = generate_recommendation(action, state, self.last_retrieved)
        return action

    def update(self, state, action, reward, next_state, done):
        self.base_agent.update(state, action, reward, next_state, done)

    def reset(self):
        self.base_agent.reset()

    def eval(self):
        if hasattr(self.base_agent, 'eval'):
            self.base_agent.eval()


def evaluate_with_rag(agent, env, n_episodes=100, seed=42):
    if hasattr(agent, 'eval'):
        agent.eval()
    np.random.seed(seed)
    results = []
    retrieval_scores = []
    for i in range(n_episodes):
        students = sample_student_from_data(1, seed=seed + i)
        init_state, gender, grade = students[0]
        state = env.reset(init_state, gender, grade)
        agent.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            if hasattr(agent, 'last_retrieved') and agent.last_retrieved:
                retrieval_scores.extend([r["score"] for r in agent.last_retrieved])
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        results.append({"total_reward": total_reward, "delta_hi": state.hi_total - (state.history[0] @ WEIGHTS)})
    return {
        "avg_reward": np.mean([r["total_reward"] for r in results]),
        "avg_delta_hi": np.mean([r["delta_hi"] for r in results]),
        "std_delta_hi": np.std([r["delta_hi"] for r in results]),
        "avg_retrieval_score": np.mean(retrieval_scores) if retrieval_scores else 0,
    }


def main():
    print("=" * 60)
    print("M3 Agentic RL + RAG 知识库对比实验")
    print("=" * 60)

    print(f"\n[1/3] 加载RAG知识库: {CHUNKS_DIR}")
    retriever = RAGRetriever(str(CHUNKS_DIR), top_k=2)
    retriever.load()

    env = HealthInterventionEnv(max_steps=5, noise_std=0.02)

    print("\n[2/3] 加载RL模型...")
    rl_agent = RLAgent(state_dim=8, n_actions=8, hidden_dim=128, n_layers=2, device=DEVICE)
    rl_agent.load(str(OUTPUT_DIR / "models" / "ppo_final.pt"))
    rl_agent.eval()

    rl_rag = RAGWrapperAgent(rl_agent, retriever)
    rule_agent = RuleAgent(rest_every=3)
    rule_rag = RAGWrapperAgent(rule_agent, retriever)

    print("\n[3/3] 对比评估（100 episodes）...")
    agents_to_eval = [
        ("RL (无RAG)", rl_agent),
        ("RL + RAG", rl_rag),
        ("Rule (无RAG)", rule_agent),
        ("Rule + RAG", rule_rag),
    ]

    all_results = {}
    for name, agent in agents_to_eval:
        print(f"  评估 {name} ...")
        if isinstance(agent, RAGWrapperAgent):
            result = evaluate_with_rag(agent, env, n_episodes=100, seed=42)
        else:
            result = evaluate_agent(agent, env, n_episodes=100, seed=42)
            result["avg_retrieval_score"] = 0
        all_results[name] = result
        print(f"    ΔHI={result['avg_delta_hi']:+.4f}  reward={result['avg_reward']:.4f}  retrieval={result.get('avg_retrieval_score', 0):.4f}")

    comparison_df = pd.DataFrame([
        {"agent": name, "avg_reward": res["avg_reward"], "avg_delta_hi": res["avg_delta_hi"],
         "std_delta_hi": res["std_delta_hi"], "avg_retrieval_score": res.get("avg_retrieval_score", 0),
         "has_rag": "+RAG" in name}
        for name, res in all_results.items()
    ])
    comparison_df.to_csv(OUTPUT_DIR / "rag_comparison.csv", index=False)

    # 案例分析
    print("\n案例分析：RAG推荐理由示例...")
    case_students = [
        ("低水平学生", np.array([-1.0, -0.8, -0.9, -0.7, -0.6, -1.0, -0.8]), "男", 1),
        ("中水平学生", np.array([0.0, 0.1, -0.2, 0.1, 0.0, -0.3, 0.1]), "女", 2),
    ]
    case_results = []
    for case_name, init_state, gender, grade in case_students:
        print(f"\n--- {case_name} ---")
        rl_rag.reset()
        state = env.reset(init_state.copy(), gender, grade)
        done = False
        step = 0
        while not done:
            action = rl_rag.select_action(state)
            rec = rl_rag.last_recommendation
            retrieved = rl_rag.last_retrieved
            next_state, reward, done, info = env.step(action)
            rl_rag.update(state, action, reward, next_state, done)
            print(f"  Step {step+1}: {ACTION_NAMES_CN[action]} | {rec[:80]}...")
            case_results.append({"case": case_name, "step": step+1, "action": ACTION_NAMES_CN[action],
                                 "recommendation": rec, "retrieval_score": retrieved[0]["score"] if retrieved else 0,
                                 "retrieval_category": retrieved[0]["category"] if retrieved else ""})
            state = next_state
            step += 1

    case_df = pd.DataFrame(case_results)
    case_df.to_csv(OUTPUT_DIR / "rag_case_studies.csv", index=False)

    with open(OUTPUT_DIR / "rag_recommendation_samples.md", "w") as f:
        f.write("# RAG 推荐理由样本\n\n")
        for _, row in case_df.iterrows():
            f.write(f"### {row['case']} - Step {row['step']}: {row['action']}\n\n")
            f.write(f"**推荐理由**: {row['recommendation']}\n\n")
            f.write(f"**检索来源**: {row['retrieval_category']} (score={row['retrieval_score']:.4f})\n\n---\n\n")

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for i, (_, row) in enumerate(comparison_df.sort_values("avg_delta_hi", ascending=False).iterrows(), 1):
        print(f"  {i}. {row['agent']:20s}  ΔHI={row['avg_delta_hi']:+.4f}  reward={row['avg_reward']:.4f}")
    rl_no = all_results["RL (无RAG)"]["avg_delta_hi"]
    rl_yes = all_results["RL + RAG"]["avg_delta_hi"]
    print(f"\nRAG对RL的ΔHI影响: {(rl_yes - rl_no):+.4f}")
    print(f"平均检索相关性: {all_results['RL + RAG']['avg_retrieval_score']:.4f}")


if __name__ == "__main__":
    main()
