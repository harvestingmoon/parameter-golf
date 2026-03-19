python3 -m venv .venv
source .venv/bin/activate
RUN_ID=smoke_test \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_v2.py