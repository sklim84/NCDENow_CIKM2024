log_path='../../_logs/cikm2024'
results_path='../../_results/cikm2024'
device=2

data='GDP_KOR'
model='rnn'
lr=1e-2
hidden_size=512
n_layers=2
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
  --n_layers $n_layers \
  --save_values

data='GDP_KOR'
model='lstm'
lr=1e-3
hidden_size=512
n_layers=4
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
  --n_layers $n_layers \
  --save_values

data='GDP_KOR'
model='gru'
lr=1e-3
hidden_size=512
n_layers=5
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
  --n_layers $n_layers \
  --save_values

data='GDP_KOR'
model='ncde'
lr=1e-2
hidden_size=16
hidden_hidden_size=128
n_layers=4
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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

data='GDP_KOR'
model='ncde_naive'
lr=1e-2
hidden_size=16
hidden_hidden_size=128
n_layers=3
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
model='rnn'
lr=1e-2
hidden_size=256
n_layers=6
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
  --n_layers $n_layers \
  --save_values

data='GDP_UK'
model='lstm'
lr=1e-3
hidden_size=256
n_layers=5
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
  --n_layers $n_layers \
  --save_values

data='GDP_UK'
model='gru'
lr=1e-2
hidden_size=128
n_layers=6
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
  --n_layers $n_layers \
  --save_values

data='GDP_UK'
model='ncde'
lr=1e-3
hidden_size=16
hidden_hidden_size=512
n_layers=5
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
model='ncde_naive'
lr=1e-2
hidden_size=16
hidden_hidden_size=64
n_layers=4
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --seed 0 \
  --model $model \
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
