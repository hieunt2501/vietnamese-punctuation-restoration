!TORCH_DISTRIBUTED_DEBUG="INFO" \
CUDA_VISIBLE_DEVICES="0, 1" \
python -m torch.distributed.launch --nproc_per_node 2 \
  ./scripts/train_electra.py \
  --model_name_or_path FPTAI/velectra-base-discriminator-cased \
  --train_data_file ./data/wiki_train_v10.csv \
  --eval_data_file ./data/wiki_eval_v10.csv \
  --do_train True \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --num_cycles 1 \
  --weight_decay 0.1 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --output_dir ./checkpoint/electra_v2_fpt \
  --overwrite_output_dir True \
  --adam_beta2 0.98 \
  --warmup_ratio 0.1 \
  --dataloader_pin_memory False \
  --dataloader_num_workers 2 \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --gradient_accumulation_step 1 \
  --logging_strategy epoch \
  --save_strategy epoch \
  --report_to tensorboard \
  --n_class 7 \
  --dataloader_shuffle True \
  --lr_scheduler_type linear \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --save_total_limit 3 \
  --fp16 True \
  --ddp_find_unused_parameters False \
  --train_nrows 50000 \
  --disable_tqdm False