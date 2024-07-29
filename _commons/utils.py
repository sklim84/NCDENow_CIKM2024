import logging
import os
import random
import warnings
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pathlib

warnings.filterwarnings('ignore')


def get_logger(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_path = os.path.join(log_path, f'{datetime.now().strftime("%Y%m%d")}.log')
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.DEBUG)

    return logger


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def save_metrics(save_path, args, arg_names, metrics, metric_names):
    columns = ['timestamp']
    values = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    for arg_name in arg_names:
        arg_value = vars(args).get(arg_name)
        columns.append(arg_name)
        values.append(arg_value)
    columns.extend(metric_names)
    values.extend(metrics)

    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path)
    else:
        df_results = pd.DataFrame(columns=columns)
    df_results.loc[len(df_results)] = values

    df_results.sort_values(by="mse", ascending=True, inplace=True)
    print(df_results)
    df_results.to_csv(save_path, index=False)


def exists_metrics(save_path, args, arg_names):
    if not os.path.exists(save_path):
        return False

    df_results = pd.read_csv(save_path)

    for index, result in df_results.iterrows():
        existence_flag = True
        for arg_name in arg_names:
            if result[arg_name] != vars(args).get(arg_name):
                existence_flag = False
                break

        if existence_flag == True:
            break

    return existence_flag


def save_pred_y(save_path, train_seq, y_seq, true_y, pred_y):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, f'true_y_{train_seq}_{y_seq}.npy'), true_y)
    np.save(os.path.join(save_path, f'pred_y_{train_seq}_{y_seq}.npy'), pred_y)


def plot_pred_y(save_path, data_name, model_name, train_seq, y_seq, true_y, pred_y):
    linewidth_base = 0.5
    linewidth_bold = 1.5
    # print(f'true_y shape: {true_y.shape}')

    x_range = range(true_y.shape[1])

    # feature plotting
    for i in range(true_y.shape[2]):
        plt.figure(figsize=(15, 9))

        # 배치 중 첫 번째 결과 출력
        plt.plot(x_range, true_y[0, :, i], label="Ground Truth", linestyle="--", linewidth=linewidth_bold)
        plt.plot(x_range, pred_y[0, :, i], label=model_name, linewidth=linewidth_base)
        plt.legend(fontsize=12, loc='lower left')
        plt.ylabel("value", fontsize=15)
        plt.xlabel("time", fontsize=15)
        plt.yticks(fontsize=10)
        # plt.xticks(ticks=np.arange(0, true_y.shape[1]), fontsize=10)
        plt.title(f'{str(data_name).upper()} (Input seq. = {train_seq}, Output seq. = {y_seq})', fontsize=15)
        plt.tight_layout()

        plt.savefig(os.path.join(save_path, f'plot_{train_seq}_{y_seq}_f{i}.png'))
        plt.close()


def moving_avg(x, kernel_size, stride):
    # x: [Batch, Input length, Channel]
    avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
    end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
    x = torch.cat([front, x, end], dim=1)
    x = avg(x.permute(0, 2, 1))
    x = x.permute(0, 2, 1)

    return x


def series_decomp(x, kernel_size):
    moving_mean = moving_avg(x, kernel_size, stride=1)
    res = x - moving_mean

    return res, moving_mean


def save_values(save_path, data_name, model_name, train_seq, y_seq, true_y, pred_y):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, f'{data_name}_{model_name}_best_test_truey_{train_seq}_{y_seq}.npy'), true_y)
    np.save(os.path.join(save_path, f'{data_name}_{model_name}_best_test_predy_{train_seq}_{y_seq}.npy'), pred_y)


def save_metrics(save_path, args, arg_names, metrics, metric_names):
    columns = ['timestamp']
    values = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    columns.extend(metric_names)
    values.extend(metrics)
    for arg_name in arg_names:
        arg_value = vars(args).get(arg_name)
        columns.append(arg_name)
        values.append(str(arg_value))

    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path)
    else:
        df_results = pd.DataFrame(columns=columns)

    df_results.loc[len(df_results)] = values
    df_results.sort_values(by='mse', ascending=True, inplace=True)
    print(df_results)
    df_results.to_csv(save_path, index=False)


def exists_metrics(save_path, args, arg_names):
    if not os.path.exists(save_path):
        return False

    df_results = pd.read_csv(save_path)

    for index, result in df_results.iterrows():
        existence_flag = True
        for arg_name in arg_names:
            result_item = result[arg_name]
            args_item = vars(args).get(arg_name)

            if result_item != args_item:
                # print(arg_name, type(result_item), result_item, type(result_item), args_item)
                existence_flag = False
                break

        if existence_flag == True:
            break

    return existence_flag


def get_logger(log_path, file_name):
    log_path = log_path

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger()
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.DEBUG)

    return logger


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def normalize(data):
    """
    Execute min-max normalization to data

    Args:
        data: target data

    Returns:
        norm_data: normalized data

    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


# StandardScaler
def standard_normalize_with_nan(data):
    for idx in range(data.shape[1]):
        da = data[:, idx]
        wo_nan_data = da[~np.isnan(da)]
        mean_data, std_data = np.mean(wo_nan_data), np.std(wo_nan_data)
        norm_da = (da - mean_data) / std_data
        data[:, idx] = norm_da
    return data


# MinMaxScaler
def minmax_normalize_with_nan(data):
    for idx in range(data.shape[1]):
        da = data[:, idx]
        wo_nan_data = da[~np.isnan(da)]
        min_data, max_data = np.min(wo_nan_data), np.max(wo_nan_data)
        numerator = da - min_data
        denominator = max_data - min_data
        norm_da = numerator / (denominator + 1e-7)
        data[:, idx] = norm_da
    return data


def time_normalize(data):
    dtime = pd.to_datetime(data)
    seconds = dtime.values.astype(np.int64) // 10 ** 9
    min_sec = min(seconds)
    seconds -= min_sec

    norm_sec = (seconds - seconds.min(axis=0)) / (seconds.max(axis=0) - seconds.min(axis=0))

    return norm_sec


def convert_date_Q_to_M(data):
    data['Date'][data['Date'].dt.month == 1] = data['Date'][data['Date'].dt.month == 1] + timedelta(days=60)
    data['Date'][data['Date'].dt.month == 4] = data['Date'][data['Date'].dt.month == 4] + timedelta(days=61)
    data['Date'][data['Date'].dt.month == 7] = data['Date'][data['Date'].dt.month == 7] + timedelta(days=62)
    data['Date'][data['Date'].dt.month == 10] = data['Date'][data['Date'].dt.month == 10] + timedelta(days=61)
    data['Date'][data['Date'].dt.day == 2] = data['Date'][data['Date'].dt.day == 2] - timedelta(days=1)
    return data


def normalize_for_dfm(data, trans_method, exception_columns=['Date']):
    for eco_idx_name in data.columns:
        if eco_idx_name in exception_columns:
            continue

        method = trans_method[eco_idx_name]
        if method == 1:  # original
            continue
        elif method == 2:  # log differencing
            # print(eco_idx_name, data[eco_idx_name])
            # print(eco_idx_name, np.log(data[eco_idx_name].shift(1)))
            data[eco_idx_name] = (np.log(data[eco_idx_name].shift(1)) - np.log(
                data[eco_idx_name])) * 100
        elif method == 3:  # original - 100
            data[eco_idx_name] = data[eco_idx_name] - 100
        elif method == 5:  # differencing
            data[eco_idx_name] = data[eco_idx_name].diff()

    return data


def set_index_to_date(data_monthly, data_quarterly):
    data_monthly['Date'] = pd.to_datetime(data_monthly['Date'])
    data_monthly.set_index('Date', inplace=True)
    data_monthly = data_monthly.asfreq(freq='MS')  # set frequency to 'M(Month)'

    data_quarterly['Date'] = pd.to_datetime(data_quarterly['Date'])
    data_quarterly['Date'] = [date - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd() for date in
                              data_quarterly['Date']]
    data_quarterly.set_index('Date', inplace=True)
    data_quarterly = data_quarterly.asfreq(freq='Q')  # set frequency to 'Q(Quarter)'

    return data_monthly, data_quarterly


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params
