CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
  --out_dir $1 \
  --data_root ./datasets \
  --batch_size 128 \
  --lr 0.06 \
  --use_eoaNet \
  --msa_scales 1 2 4 \
  --eog_beta 0.5 \
