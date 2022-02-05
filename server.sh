CUDA_LAUNCH_BLOCKING=1
nohup python3 scripts/dialog.py data/sample/bin/ \
 --path /home/ubuntu/Platform/data/model/checkpoint_best.pt \
 --beam 30 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model /home/ubuntu/Platform/data/dicts/sp_oall_32k.model \
 --no-repeat-ngram-size 3 \
 --nbest 30 \
 --sampling \
 --sampling-topp 0.9 \
 --temperature 1.0 \
 --show-nbest 30 > out.log 2> error.log &