RUN_ID=smoke_test \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=50 \
VAL_BATCH_SIZE=65536 \
python3 train_gpt_v2.py

#  sed -i 's/\r//' run.sh  