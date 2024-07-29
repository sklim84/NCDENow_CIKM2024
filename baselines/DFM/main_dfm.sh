log_path='../../_logs/cikm2024'
results_path='../../_results/cikm2024'

data='GDP_KOR'
n_factors=2
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --model 'dfm' \
  --n_factors $n_factors \
  --data $data \
  --train_seq 15 \
  --y_seq 1 \
  --save_values

data='GDP_UK'
n_factors=8
python3 -u main.py \
  --log_path $log_path \
  --results_path $results_path \
  --model 'dfm' \
  --n_factors $n_factors \
  --data $data \
  --train_seq 15 \
  --y_seq 1 \
  --save_values
