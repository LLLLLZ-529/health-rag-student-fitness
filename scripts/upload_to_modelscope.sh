#!/usr/bin/env bash
# upload_to_modelscope.sh
# Upload LoRA adapter weights to ModelScope.
#
# Prerequisites:
#   pip install modelscope
#   modelscope login   # enter your ModelScope API token
#
# Usage:
#   bash scripts/upload_to_modelscope.sh /path/to/health_rag_final_package/models
#
# This uploads two LoRA adapters fine-tuned on Qwen2.5-7B-Instruct:
#   - sft_best_model  -> anne118/health-rag-sft-7b
#   - dpo_model       -> anne118/health-rag-dpo-7b
#
# The namespace below (anne118) is your ModelScope username. Change it if needed.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "错误：缺少模型目录参数"
  echo "正确用法：bash scripts/upload_to_modelscope.sh /Users/annedl/Documents/health_rag_final_package/models"
  exit 1
fi

MODELS_DIR="$1"
NAMESPACE="${MODELSCOPE_NAMESPACE:-anne118}"

if [ ! -d "${MODELS_DIR}/sft_best_model" ] || [ ! -d "${MODELS_DIR}/dpo_model" ]; then
  echo "错误：在 ${MODELS_DIR} 下未找到 sft_best_model/ 或 dpo_model/ 目录"
  exit 1
fi

# 创建模型仓库（已存在时 --exist-ok 不报错）
echo "=== Ensuring model repositories exist ==="
modelscope create "anne118/health-rag-sft-7b" \
  --repo-type model --visibility public --license apache-2.0 --exist-ok \
  --description "LoRA SFT adapter for Qwen2.5-7B-Instruct, student health RAG demo" || true
modelscope create "anne118/health-rag-dpo-7b" \
  --repo-type model --visibility public --license apache-2.0 --exist-ok \
  --description "LoRA DPO adapter for Qwen2.5-7B-Instruct, student health RAG demo" || true

echo ""
echo "=== Uploading SFT adapter ==="
modelscope upload "${NAMESPACE}/health-rag-sft-7b" "${MODELS_DIR}/sft_best_model" \
  --repo-type model \
  --commit-message "Upload SFT LoRA adapter (Qwen2.5-7B-Instruct)"

echo ""
echo "=== Uploading DPO adapter ==="
modelscope upload "${NAMESPACE}/health-rag-dpo-7b" "${MODELS_DIR}/dpo_model" \
  --repo-type model \
  --commit-message "Upload DPO LoRA adapter (Qwen2.5-7B-Instruct)"

echo ""
echo "=== Done ==="
echo "SFT: https://modelscope.cn/models/${NAMESPACE}/health-rag-sft-7b"
echo "DPO: https://modelscope.cn/models/${NAMESPACE}/health-rag-dpo-7b"
