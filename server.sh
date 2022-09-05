DATE=`date "+%Y%m%d-%H%M"`
CUDA_LAUNCH_BLOCKING=1
python3 scripts/dialog.py data/sample/bin/ \
 --path /home/ubuntu/Platform/checkpoints/checkpoint_best.pt \
 --beam 10 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model /home/ubuntu/Platform/data/dicts/sp_oall_32k.model \
 --no-repeat-ngram-size 3 \
 --nbest 10 \
 --sampling \
 --sampling-topp 0.9 \
 --temperature 1.0 \
 --show-nbest 15 > log/out_${DATE}.log 2> log/error_${DATE}.log