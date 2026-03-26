set -euo pipefail

# ── Launcher setup ─────────────────────────────────────────────────────────────
NPROC="${NPROC:-1}"
LARGE_GPU="${LARGE_GPU:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
if [ "$NPROC" -gt 1 ]; then
    LAUNCHER=("$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" --standalone)
else
    LAUNCHER=("$PYTHON")
fi

# ── Run ID ─────────────────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-baseline_int5_ppm_$(date +%Y%m%d_%H%M)}"

# ── Architecture ───────────────────────────────────────────────────────────────
NUM_LAYERS="${NUM_LAYERS:-11}"
MODEL_DIM="${MODEL_DIM:-512}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
MLP_HIDDEN="${MLP_HIDDEN:-1536}"

# SwiGLU + PowerMLP gate (relu^n + silu) — best bpb from ablations
USE_SWIGLU="${USE_SWIGLU:-1}"
USE_POWER_MLP_SWIGLU="${USE_POWER_MLP_SWIGLU:-1}"
POWER_MLP_REPU_ORDER="${POWER_MLP_REPU_ORDER:-2}"

# NoPE: fraction of blocks that skip RoPE (0.25 = every 4th block)
NOPE_RATIO="${NOPE_RATIO:-0.25}"
NOPE_MODE="${NOPE_MODE:-block}"    # "block" (whole block NoPE) | "head" (per-head)

# Tied embeddings + logit softcap
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}"

# ── Training schedule ──────────────────────────────────────────────────────────
SEED="${SEED:-1337}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
ITERATIONS="${ITERATIONS:-200}"           # set to 20000 for full run
WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
GRAD_ACCUM_TARGET="${GRAD_ACCUM_TARGET:-8}"

# ── Validation + logging ───────────────────────────────────────────────────────
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-50}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-64}"
SKIP_SLIDING_EVAL="${SKIP_SLIDING_EVAL:-1}"   # 0 = run sliding-window eval at end

# ── Learning rates ─────────────────────────────────────────────────────────────
EMBED_LR="${EMBED_LR:-0.6}"
HEAD_LR="${HEAD_LR:-0.008}"
TIED_EMBED_LR="${TIED_EMBED_LR:-0.050}"
MATRIX_LR="${MATRIX_LR:-0.04}"
SCALAR_LR="${SCALAR_LR:-0.04}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# ── Optimizer ──────────────────────────────────────────────────────────────────
USE_ADAMUON="${USE_ADAMUON:-1}"
ADAMUON_BETA2="${ADAMUON_BETA2:-0.92175}"
ADAMUON_EPS="${ADAMUON_EPS:-1e-8}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"

# ── EMA (exponential moving average of weights) ────────────────────────────────
EMA_ENABLED="${EMA_ENABLED:-1}"
EMA_DECAY="${EMA_DECAY:-0.997}"

# ── SWA (stochastic weight averaging) ─────────────────────────────────────────
SWA_ENABLED="${SWA_ENABLED:-1}"
SWA_EVERY="${SWA_EVERY:-10}"

# ── Architecture options ───────────────────────────────────────────────────────
LN_SCALE="${LN_SCALE:-1}"                    # 1/sqrt(layer_idx+1) input scaling
FOCAL_GAMMA="${FOCAL_GAMMA:-0.45}"           # focal loss exponent; 0=plain CE
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"

# Backout: subtract mid-network shortcut from final hidden state
BACKOUT_ENABLED="${BACKOUT_ENABLED:-1}"
BACKOUT_LAMBDA_INIT="${BACKOUT_LAMBDA_INIT:-0.24}"
BACKOUT_LAYER="${BACKOUT_LAYER:--1}"         # -1 = auto (mid-network)

# Backward-looking LoRA
USE_LORA="${USE_LORA:-1}"
LORA_RANK="${LORA_RANK:-16}"

# ── Quantization-aware training ────────────────────────────────────────────────
# LATE_QAT fires QAT when LR scale < threshold (0.5 = mid-warmdown)
LATE_QAT="${LATE_QAT:-1}"
LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.50}"
QAT_ENABLED="${QAT_ENABLED:-0}"             # 1 = enable QAT from step 0 (short runs)
INT6_LAST_N="${INT6_LAST_N:-0}"             # last N blocks use int6; rest use int5
PRUNE_PCT="${PRUNE_PCT:-0.03}"              # magnitude pruning before quantization

# ── N-gram / PPM mixer ─────────────────────────────────────────────────────────
# MIXER_EVAL=1: run mixer-only sliding eval (no TTT) after quantization roundtrip.
#               Shows mixer's isolated bpb gain vs plain neural baseline.
# USE_MIXER=1:  also blend mixer inside TTT scoring (when TTT_ENABLED=1).
# MIXER_TYPE:   "ngram" = BackoffNgramMixer (fast, hashed, orders 2–7)
#               "ppm"   = ByteLevelPPM-C (slower, better escape probabilities)
# MIXER_ETA:    learning rate for online mixer weight update (currently unused)
# PPM_ALPHA:    base blending weight for mixer vs neural (rest is entropy-adaptive)
MIXER_EVAL="${MIXER_EVAL:-1}"               # default ON: see mixer effect without TTT
USE_MIXER="${USE_MIXER:-1}"
MIXER_TYPE="${MIXER_TYPE:-ppm}"             # default: PPM-C (better than n-gram)
MIXER_ETA="${MIXER_ETA:-0.1}"
PPM_ALPHA="${PPM_ALPHA:-0.85}"

# ── TTT (chunked score-first test-time training) ───────────────────────────────
TTT_ENABLED="${TTT_ENABLED:-1}"
TTT_EPOCHS="${TTT_EPOCHS:-1}"
TTT_LR="${TTT_LR:-3e-5}"
TTT_BATCH_SEQS="${TTT_BATCH_SEQS:-32}"
TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-1}" # unfreeze only last N blocks during TTT
TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-1048576}"  # tokens per scored/trained chunk (1M)
TTT_TEMPERATURE="${TTT_TEMPERATURE:-0.98}"       # temperature scaling for scoring phase
TTT_OPTIMIZER="${TTT_OPTIMIZER:-adamw}"
BYTE_WEIGHTED_TTT="${BYTE_WEIGHTED_TTT:-1}" # weight TTT loss by token byte length
POLYAK_DECAY="${POLYAK_DECAY:-0.998}"       # Polyak averaging decay for scoring weights

# ── Optional extras (off by default) ──────────────────────────────────────────
USE_BIGRAM="${USE_BIGRAM:-0}"               # BigramHashEmbedding
BIGRAM_BUCKETS="${BIGRAM_BUCKETS:-2048}"
BIGRAM_DIM="${BIGRAM_DIM:-128}"
USE_SHARED_VALUE_EMB="${USE_SHARED_VALUE_EMB:-0}"
VE_ENABLED="${VE_ENABLED:-0}"
VE_DIM="${VE_DIM:-128}"
VE_LAYERS="${VE_LAYERS:-9,10}"
USE_ATTNRES="${USE_ATTNRES:-0}"

# ── Attention backend ──────────────────────────────────────────────────────────
# FLASH_ATTN=1: use Flash Attention 3 (requires Hopper / flash_attn_interface).
# FLASH_ATTN=0: fall back to torch scaled_dot_product_attention (works on any GPU).
FLASH_ATTN="${FLASH_ATTN:-0}"   # default OFF for RTX 4060 compatibility

# TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=524288 NPROC=8 FLASH_ATTN=1

# ── Log file ───────────────────────────────────────────────────────────────────
LOG_FILE="${LOG_FILE:-logs/${RUN_ID}.txt}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=== run_baseline_new.sh ==="
echo "  NPROC=${NPROC}  LARGE_GPU=${LARGE_GPU}"
echo "  RUN_ID=${RUN_ID}"
echo "  ITERATIONS=${ITERATIONS}  MAX_WALLCLOCK=${MAX_WALLCLOCK_SECONDS}s"
echo "  MIXER_TYPE=${MIXER_TYPE}  USE_MIXER=${USE_MIXER}  TTT_ENABLED=${TTT_ENABLED}  FLASH_ATTN=${FLASH_ATTN}"
echo "  LOG -> ${LOG_FILE}"
echo ""

RUN_ID="$RUN_ID" \
SEED="$SEED" \
NUM_LAYERS="$NUM_LAYERS"            MODEL_DIM="$MODEL_DIM" \
NUM_HEADS="$NUM_HEADS"              NUM_KV_HEADS="$NUM_KV_HEADS" \
MLP_HIDDEN="$MLP_HIDDEN" \
USE_SWIGLU="$USE_SWIGLU"           USE_POWER_MLP_SWIGLU="$USE_POWER_MLP_SWIGLU" \
POWER_MLP_REPU_ORDER="$POWER_MLP_REPU_ORDER" \
NOPE_RATIO="$NOPE_RATIO"           NOPE_MODE="$NOPE_MODE" \
TIE_EMBEDDINGS="$TIE_EMBEDDINGS"   LOGIT_SOFTCAP="$LOGIT_SOFTCAP" \
TRAIN_BATCH_TOKENS="$TRAIN_BATCH_TOKENS" \
TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
ITERATIONS="$ITERATIONS"           WARMDOWN_ITERS="$WARMDOWN_ITERS" \
WARMUP_STEPS="$WARMUP_STEPS"       LR_WARMUP_STEPS="$LR_WARMUP_STEPS" \
MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
GRAD_ACCUM_TARGET="$GRAD_ACCUM_TARGET" \
VAL_LOSS_EVERY="$VAL_LOSS_EVERY"   VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
EVAL_STRIDE="$EVAL_STRIDE"         EVAL_BATCH_SEQS="$EVAL_BATCH_SEQS" \
SKIP_SLIDING_EVAL="$SKIP_SLIDING_EVAL" \
EMBED_LR="$EMBED_LR"               HEAD_LR="$HEAD_LR" \
TIED_EMBED_LR="$TIED_EMBED_LR"     MATRIX_LR="$MATRIX_LR" \
SCALAR_LR="$SCALAR_LR"             GRAD_CLIP_NORM="$GRAD_CLIP_NORM" \
WEIGHT_DECAY="$WEIGHT_DECAY" \
USE_ADAMUON="$USE_ADAMUON"         ADAMUON_BETA2="$ADAMUON_BETA2" \
ADAMUON_EPS="$ADAMUON_EPS"         MUON_MOMENTUM="$MUON_MOMENTUM" \
BETA1="$BETA1"                     BETA2="$BETA2" \
EMA_ENABLED="$EMA_ENABLED"         EMA_DECAY="$EMA_DECAY" \
SWA_ENABLED="$SWA_ENABLED"         SWA_EVERY="$SWA_EVERY" \
LN_SCALE="$LN_SCALE" \
FOCAL_GAMMA="$FOCAL_GAMMA"         LABEL_SMOOTHING="$LABEL_SMOOTHING" \
BACKOUT_ENABLED="$BACKOUT_ENABLED" BACKOUT_LAMBDA_INIT="$BACKOUT_LAMBDA_INIT" \
BACKOUT_LAYER="$BACKOUT_LAYER" \
USE_LORA="$USE_LORA"               LORA_RANK="$LORA_RANK" \
LATE_QAT="$LATE_QAT"               LATE_QAT_THRESHOLD="$LATE_QAT_THRESHOLD" \
QAT_ENABLED="$QAT_ENABLED" \
INT6_LAST_N="$INT6_LAST_N"         PRUNE_PCT="$PRUNE_PCT" \
MIXER_EVAL="$MIXER_EVAL" \
USE_MIXER="$USE_MIXER"             MIXER_TYPE="$MIXER_TYPE" \
MIXER_ETA="$MIXER_ETA"             PPM_ALPHA="$PPM_ALPHA" \
TTT_ENABLED="$TTT_ENABLED"         TTT_EPOCHS="$TTT_EPOCHS" \
TTT_LR="$TTT_LR"                   TTT_BATCH_SEQS="$TTT_BATCH_SEQS" \
TTT_FREEZE_BLOCKS="$TTT_FREEZE_BLOCKS" \
TTT_CHUNK_TOKENS="$TTT_CHUNK_TOKENS" \
TTT_TEMPERATURE="$TTT_TEMPERATURE" TTT_OPTIMIZER="$TTT_OPTIMIZER" \
BYTE_WEIGHTED_TTT="$BYTE_WEIGHTED_TTT" \
POLYAK_DECAY="$POLYAK_DECAY" \
USE_BIGRAM="$USE_BIGRAM"           BIGRAM_BUCKETS="$BIGRAM_BUCKETS" \
BIGRAM_DIM="$BIGRAM_DIM" \
USE_SHARED_VALUE_EMB="$USE_SHARED_VALUE_EMB" \
VE_ENABLED="$VE_ENABLED"           VE_DIM="$VE_DIM" \
VE_LAYERS="$VE_LAYERS" \
USE_ATTNRES="$USE_ATTNRES" \
FLASH_ATTN="$FLASH_ATTN" \
LARGE_GPU="$LARGE_GPU" \
"${LAUNCHER[@]}" train_gpt_baseline.py 2>&1 | tee "$LOG_FILE"







# NPROC=8 \
# LARGE_GPU=1 \
# FLASH_ATTN=1 \
# \
# RUN_ID="h100_int5_ppm_$(date +%Y%m%d_%H%M)" \
# \
# NUM_LAYERS=11 \
# MODEL_DIM=512 \
# NUM_HEADS=8 \
# NUM_KV_HEADS=4 \
# MLP_HIDDEN=1536 \
# \
# TRAIN_BATCH_TOKENS=524288 \
# TRAIN_SEQ_LEN=2048 \
# ITERATIONS=20000 \
# WARMDOWN_ITERS=3000 \
# WARMUP_STEPS=200 \
# LR_WARMUP_STEPS=200 \
# MAX_WALLCLOCK_SECONDS=360 \
# GRAD_ACCUM_TARGET=1 \
# \
# VAL_LOSS_EVERY=500 \
# VAL_BATCH_SIZE=524288 \
# TRAIN_LOG_EVERY=100 \
# SKIP_SLIDING_EVAL=1 \
# EVAL_STRIDE=512 \
# EVAL_BATCH_SEQS=128 \
# \
# EMA_ENABLED=1 \
# EMA_DECAY=0.997 \
# SWA_ENABLED=1 \
# SWA_EVERY=10 \
# \
# LATE_QAT=1 \
# LATE_QAT_THRESHOLD=0.50 \
# QAT_ENABLED=0 \
# PRUNE_PCT=0.03 \
# INT6_LAST_N=0 \
# \
# MIXER_EVAL=1 \
# USE_MIXER=1 \
# MIXER_TYPE=ppm \
# PPM_ALPHA=0.85 \
# \
# TTT_ENABLED=0 \
# \
# bash run_baseline_new.sh