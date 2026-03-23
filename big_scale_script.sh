# ── Big scale: 8xH100 SXM ─────────────────────────────────────────────────────
# Usage: NPROC=8 LARGE_GPU=1 bash big_scale_script.sh
NPROC=${NPROC:-8}
LARGE_GPU=${LARGE_GPU:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
if [ "$NPROC" -gt 1 ]; then
    LAUNCHER=("$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" --standalone)
else
    LAUNCHER=("$PYTHON")
fi

RUN_ID=big_scale_524288_new_main \
SEED_ID=1337 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=1536 \
MLP_INT6=1 \
USE_SWIGLU=1 USE_BAYES_MLP=1 \
NOPE_RATIO=0.25 NOPE_MODE=block \
USE_ALIBI=0 \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 \
ITERATIONS=20000 WARMDOWN_ITERS=3000 WARMUP_STEPS=20 LR_WARMUP_STEPS=20 \
VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=786432 \
TRAIN_LOG_EVERY=250 \
QUANT_AUTO_BUDGET_MB=0 MAX_WALLCLOCK_SECONDS=600 \
LARGE_GPU="${LARGE_GPU}" \
EMA_ENABLED=0 EMA_DECAY=0.999 \
LN_SCALE=1 \
BACKOUT_ENABLED=1 BACKOUT_LAMBDA_INIT=0.12 BACKOUT_LAYER=-1 \
USE_ADAMUON=1 ADAMUON_BETA2=0.92 \
USE_ATTNRES=0 \
ATTNRES_BLOCK_SIZE=4 \
USE_BIGRAM=1 BIGRAM_BUCKETS=2048 BIGRAM_DIM=128 \
USE_SHARED_VALUE_EMB=1 \
USE_LORA=1 LORA_RANK=16 \
GRAD_CLIP_NORM=0.1 \
FOCAL_GAMMA=0.6 \
TTT_ENABLED=1 \
TTT_EPOCHS=10 \
TTT_LR=1e-4 \
TTT_BATCH_SEQS=32 \
TTT_FREEZE_BLOCKS=0 \
LATE_QAT=1 \
LATE_QAT_THRESHOLD=0.15 \
SWA_ENABLED=1 \
SWA_EVERY=200 \
"${LAUNCHER[@]}" train_gpt_smear_attn.py 


\
`# ── Multi-Token Prediction (MTP) ─────────────────────────────────────────────` \
`# Adds N auxiliary heads that predict k+1..k+N future tokens from the same`    \
`# hidden state. Acts as a regularizer; heads are excluded from the checkpoint.` \

\
`# ── STE int6 Quantization-Aware Training (QAT) ───────────────────────────────` \
`# Applies a per-row int6 Straight-Through Estimator to all CastedLinear weights`\
`# once the LR cosine scale drops below LATE_QAT_THRESHOLD, bridging the`        \
`# train/eval quantization gap in the final phase of training.`                  \

\
`# ── Stochastic Weight Averaging (SWA) ────────────────────────────────────────`\
`# Accumulates a running average of model snapshots every SWA_EVERY steps and`   \
`# replaces the weights with the average before final eval/quantization.`        \
`# Usually lowers val BPB by ~0.001-0.003 at no extra training cost.`           \