
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
out_dir=$1
python -u experiment.py \
  --out_dir ${out_dir} \
  --is_training 1 \
  --root_path ./datasets/tsf/dataset/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1.log

