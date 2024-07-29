import os
import sys
import warnings

parent_path = (os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(parent_path)
parent_parent_path = (os.path.dirname(os.path.abspath(parent_path)))
sys.path.append(parent_parent_path)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from _commons.loss import loss_mse
from _commons.utils import save_metrics, exists_metrics, save_values
from _datasets import preprocessor
from config import get_config
from model import DynamicFactorModel

warnings.filterwarnings('ignore')

args = get_config()
print(args)

# check existence of experiments results
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
metric_save_path = os.path.abspath(os.path.join(args.results_path, f'{args.model}_metrics.csv'))
arg_names = ['model', 'data', 'train_seq', 'y_seq', 'n_factors', 'factor_orders']
# if exists_metrics(metric_save_path, args, arg_names):
#     print(f'There exist experiments results! - {args}')
#     sys.exit()

if 'KOR' in args.data:
    target = 'GDP'
elif 'UK' in args.data:
    target = 'GDPP'
else:
    target = ''

trainX, trainy, validX, validy, testX, testy, group_info, num_monthly \
    = preprocessor.for_DFM(args.train_seq, args.y_seq, data=args.data, target=target)
# reconstruct train data for DFM model fitting
trainX = pd.concat(trainX).drop_duplicates().sort_index()

model = DynamicFactorModel(data=trainX, idx_target=target, num_monthly=num_monthly, factors=group_info,
                           factor_orders=args.factor_orders)

# train : fit DFM model
model.fit()

# valid
valid_yp = np.array([])
for X in validX:
    y_p = model.apply_and_forecast(data=X, forcast_steps=args.y_seq)
    valid_yp = np.append(valid_yp, y_p)

valid_yp = np.expand_dims(valid_yp, axis=1)
validy = validy.squeeze(-1)
valid_mse = loss_mse(valid_yp, validy)
valid_mape = mean_absolute_percentage_error(validy, valid_yp) * 100
print('Valid MSE: {:.8f} | MAPE: {:.8f}'.format(valid_mse, valid_mape))

# test
test_yp = np.array([])
for X in testX:
    y_p = model.apply_and_forecast(data=X, forcast_steps=args.y_seq)
    # print(y_p)
    test_yp = np.append(test_yp, y_p)

test_yp = np.expand_dims(test_yp, axis=1)
testy = testy.squeeze(-1)
test_mse = loss_mse(test_yp, testy)
test_mape = mean_absolute_percentage_error(testy, test_yp) * 100
print('Test MSE: {:.8f} | MAPE: {:.8f}'.format(test_mse, test_mape))

# save_results
metric_names = ['mse', 'mape']
metrics = [test_mse, test_mape]
save_metrics(metric_save_path, args, arg_names, metrics, metric_names)

if args.save_values:
    values_save_path = os.path.abspath(os.path.join(args.results_path, 'values'))
    save_values(values_save_path, args.data, args.model, args.train_seq, args.y_seq, testy, test_yp)
