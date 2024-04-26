python finetune.py \
    --pretrain_ckpt checkpoint/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/Scientific \
    --num_train_epochs 128 \
    --batch_size 32 \
    --device 0 \
    --fp16 \
    --finetune_negative_sample_size -1 \
    --max_item_num 4