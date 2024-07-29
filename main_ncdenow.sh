log_path='./_logs/cikm2024'
results_path='./_results/cikm2024'
device=0

data='GDP_KOR'
n_factors=2
cde_type='MLPCDEFunc'
lr=1e-2
hidden_size=16
hidden_hidden_size=128
n_layers=1
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --n_factors $n_factors \
  --seed 0 \
  --model 'ncdenow' \
  --cde_type $cde_type \
  --data $data \
  --train_seq 15 \
  --y_seq 1 \
  --epochs 1000 \
  --device $device \
  --lr $lr \
  --batch 128 \
  --weight_decay 1e-5 \
  --early_stopping_patience 5 \
  --hidden_size $hidden_size \
  --hidden_hidden_size $hidden_hidden_size \
  --n_layers $n_layers \
  --save_values

data='GDP_UK'
n_factors=8
cde_type='MLPCDEFunc'
lr=1e-3
hidden_size=512
hidden_hidden_size=256
n_layers=1
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --n_factors $n_factors \
  --seed 0 \
  --model 'ncdenow' \
  --cde_type $cde_type \
  --data $data \
  --train_seq 15 \
  --y_seq 1 \
  --epochs 1000 \
  --device $device \
  --lr $lr \
  --batch 128 \
  --weight_decay 1e-5 \
  --early_stopping_patience 5 \
  --hidden_size $hidden_size \
  --hidden_hidden_size $hidden_hidden_size \
  --n_layers $n_layers \
  --save_values
