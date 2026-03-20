# 1) BASELINE smoke test (12L, no MHC)
# RUN_ID=baseline_12L \
# ITERATIONS=200 TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 VAL_BATCH_SIZE=65536 EVAL_STRIDE=512 EVAL_BATCH_SEQS=8 \
# NUM_LAYERS=12 MLP_HIDDEN=512 QUANT_AUTO_BUDGET_MB=15.0 \
# python3 train_gpt_v2.py 2>&1 | tee logs/baseline_12L.txt

# 2) DEEPER baseline (20L, wider MLP, no MHC) (3:1 ratio)
# RUN_ID=deeper_20L \
# ITERATIONS=200 TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 VAL_BATCH_SIZE=65536 EVAL_STRIDE=512 EVAL_BATCH_SEQS=8 \
# NUM_LAYERS=20 MLP_HIDDEN=768 QUANT_AUTO_BUDGET_MB=15.0 \
# python3 train_gpt_v2_mhc.py

# 2b) 1:1 GDN:attn ratio — best architecture (defaults baked into train_gpt_mamba_delta.py)
# RUN_ID=attn_every2_1to1 \
# ATTN_EVERY=2 \
# ITERATIONS=200 TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 VAL_BATCH_SIZE=65536 EVAL_STRIDE=512 EVAL_BATCH_SEQS=8 \
# NUM_LAYERS=20 MLP_HIDDEN=768 QUANT_AUTO_BUDGET_MB=15.0 \
# python3 train_gpt_v2_mhc.py 2>&1 | tee logs/attn_every2_1to1.txt

# ── ACTIVE: mamba_delta — 1:1 ratio, stride=64, batch=32, no MHC ─────────────
# RUN_ID=mamba_delta_1to1 \
# ITERATIONS=200 TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 VAL_BATCH_SIZE=65536 \
# QUANT_AUTO_BUDGET_MB=15.0 MAX_WALLCLOCK_SECONDS=600 \
# python3 train_gpt_mamba_delta.py 2>&1 | tee logs/mamba_delta_1to1.txt

# ── smear_gdn_v1 — SmearGate + MuonWD + int6 MLP (20L, mlp=768) ──────────────
# RUN_ID=smear_gdn_v1 \
# ITERATIONS=200 TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 VAL_BATCH_SIZE=65536 \
# QUANT_AUTO_BUDGET_MB=15.0 MAX_WALLCLOCK_SECONDS=600 \
# python3 train_gpt_smear_gdn.py 2>&1 | tee logs/smear_gdn_v1.txt

# ── ACTIVE: wide_mlp — 11L, dim=512, MLP=1536 (3×), int6 MLP, GDN backend ────
# RUN_ID=wide_mlp_11L \
# NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=1536 \
# ATTN_EVERY=2 MLP_INT6=1 SSM_BACKEND=gdn \
# ITERATIONS=200 TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 VAL_BATCH_SIZE=65536 \
# QUANT_AUTO_BUDGET_MB=15.0 MAX_WALLCLOCK_SECONDS=600 \
# EVAL_STRIDE=0 \
# python3 train_gpt_smear_gdn_mamba.py


#python3 smear_gate.py

#  sed -i 's/\r//' run.sh
# MODEL_DIM=512 \
# ATTN_EVERY=4 \
# QUANT_AUTO_BUDGET_MB=15.0 \
# ITERATIONS=200 \
# TRAIN_BATCH_TOKENS=65536 \
# VAL_LOSS_EVERY=50 \
# VAL_BATCH_SIZE=65536 \
# EVAL_STRIDE=512 \
# EVAL_BATCH_SEQS=8 \
# python3 train_gpt_v2.py 2>&1 | tee logs/wider_deeper_20L.txt