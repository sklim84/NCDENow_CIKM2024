import json
import os
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import _commons.utils as utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm

import torchcde
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from enum import Enum
import torch
import re


# for ncde_naive
def default(train_seq, y_seq, NCDE=True, use_diff_norm=False, use_min_max_norm=True,
            train_split=0.7, valid_split=0.15, test_split=0.15,
            data='GDP_UK', target='GDPP'):
    here = pathlib.Path(__file__).resolve().parent
    folder_loc = f'_processed_data/{data}/default'
    checkFile_loc = f"{data}_trainX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    if os.path.exists(here / folder_loc / checkFile_loc):
        return load_processed_data(folder_loc, train_seq, y_seq,
                                   NCDE, use_diff_norm, use_min_max_norm, train_split,
                                   valid_split, test_split)

    data_monthly = pd.read_csv(here / f'{data}_monthly.csv')
    if 'miss' in data:
        data = re.sub('_miss_r[0-9]+', '', data)
    data_quarterly = pd.read_csv(here / f'{data}_quarterly.csv')

    data_monthly['Date'] = pd.to_datetime(data_monthly['Date'])
    data_quarterly['Date'] = pd.to_datetime(data_quarterly['Date'])

    data_quarterly = utils.convert_date_Q_to_M(data_quarterly)
    data_combine = data_quarterly.merge(data_monthly, on='Date', how='right')

    dtime = np.expand_dims(utils.time_normalize(data_combine["Date"]), axis=1)
    data_X = utils.minmax_normalize_with_nan(data_combine.drop(['Date'], axis=1).to_numpy(dtype='float32'))
    data_y = np.expand_dims(data_combine[target].values, axis=1)
    # add GDP as the last feature (to accurately order the sequence)
    data_X = np.concatenate([data_X, data_y], axis=1)

    full_seq = train_seq + y_seq
    seq_data = None

    timestep = np.arange(2, len(data_X) - full_seq + 1, step=3)
    for i in timestep:
        if NCDE:
            new_seq = np.concatenate([dtime[i:i + full_seq], data_X[i:i + full_seq]], axis=1)
        else:
            new_seq = data_X[i:i + full_seq]
        new_seq = np.expand_dims(new_seq, axis=0)

        if seq_data is None:
            seq_data = new_seq
        else:
            seq_data = np.concatenate([seq_data, new_seq], axis=0)

    total_num = seq_data.shape[0]

    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
    trainX, trainy = seq_data[:train_len, :train_seq, :-1], seq_data[:train_len, train_seq:, -1]
    validX, validy = seq_data[train_len:valid_len, :train_seq, :-1], seq_data[train_len:valid_len, train_seq:, -1]
    testX, testy = seq_data[valid_len:, :train_seq, :-1], seq_data[valid_len:, train_seq:, -1]

    save_processed_data(folder_loc, train_seq, y_seq, NCDE, use_diff_norm, use_min_max_norm, train_split, valid_split,
                        test_split, trainX, trainy, validX, validy, testX, testy)

    return trainX, trainy, validX, validy, testX, testy


def default_FXR(train_seq, y_seq, NCDE=True, use_diff_norm=False, use_min_max_norm=True,
                train_split=0.7, valid_split=0.15, test_split=0.15,
                data='FXR_ETRI', target='원달러환율'):
    here = pathlib.Path(__file__).resolve().parent

    here = pathlib.Path(__file__).resolve().parent
    folder_loc = f'_processed_data/{data}/default'
    checkFile_loc = f"{data}_trainX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    if os.path.exists(here / folder_loc / checkFile_loc):
        return load_processed_data(folder_loc, train_seq, y_seq,
                                   NCDE, use_diff_norm, use_min_max_norm, train_split,
                                   valid_split, test_split)

    df_data = pd.read_csv(here / f'{data}.csv')
    dtime = np.expand_dims(utils.time_normalize(df_data["Date"]), axis=1)
    data_X = utils.normalize(df_data.drop(['Date'], axis=1).to_numpy(dtype='float32'))
    data_y = np.expand_dims(df_data[target].values, axis=1) / 1000  # 단위: 천원

    total_seq_data, total_seq_y = None, None
    timestep = range(len(data_X) - train_seq - y_seq + 1)

    for i in timestep:
        if NCDE:
            seq_data = np.concatenate([dtime[i:i + train_seq], data_X[i:i + train_seq]], axis=1)
        else:
            seq_data = data_X[i:i + train_seq]
        seq_data = np.expand_dims(seq_data, axis=0)

        if total_seq_data is None:
            total_seq_data = seq_data
        else:
            total_seq_data = np.concatenate([total_seq_data, seq_data], axis=0)

        seq_y = np.expand_dims(data_y[i + train_seq:i + train_seq + y_seq], axis=0)
        if total_seq_y is None:
            total_seq_y = seq_y
        else:
            total_seq_y = np.concatenate([total_seq_y, seq_y], axis=0)

    total_num = total_seq_data.shape[0]

    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
    trainX, trainy = total_seq_data[:train_len], total_seq_y[:train_len]
    validX, validy = total_seq_data[train_len:valid_len], total_seq_y[train_len:valid_len]
    testX, testy = total_seq_data[valid_len:], total_seq_y[valid_len:]

    save_processed_data(folder_loc, train_seq, y_seq, NCDE, use_diff_norm, use_min_max_norm, train_split, valid_split,
                        test_split, trainX, trainy, validX, validy, testX, testy)

    return trainX, trainy, validX, validy, testX, testy


def with_impute(train_seq, y_seq, NCDE=True, use_diff_norm=True, use_min_max_norm=True,
                train_split=0.7, valid_split=0.15, test_split=0.15,
                data='GDP_UK', target='GDPP'):
    here = pathlib.Path(__file__).resolve().parent
    folder_loc = f'_processed_data/{data}/imputed'
    checkFile_loc = f'{data}_trainX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy'

    if os.path.exists(here / folder_loc / checkFile_loc):
        return load_processed_data(folder_loc, train_seq, y_seq, NCDE, use_diff_norm, use_min_max_norm, train_split,
                                   valid_split,
                                   test_split)

    # load monthly and quarterly data respectively
    data_monthly_total = pd.read_csv(here / f'{data}_monthly.csv')  # 28
    if 'miss' in data:
        data = re.sub('_miss_r[0-9]+', '', data)
    data_quarterly_total = pd.read_csv(here / f'{data}_quarterly.csv')  # 5

    seq_month, seq_quarter = train_seq, int(train_seq / 3)
    dtime = np.expand_dims(utils.time_normalize(data_monthly_total['Date']), axis=1)
    data_y = np.expand_dims(data_quarterly_total[target].values, axis=1)

    # feature group information for DFM
    with open(here / f'{data}_info.json', 'r') as f:
        GDP_info = json.load(f)
        group_info = GDP_info["Group"]
        trans_method = GDP_info["Trans"]

    seq_data = None
    time_step = range(len(data_quarterly_total) - (seq_quarter + y_seq) + 1)

    for i in tqdm(time_step):
        idx_month, idx_quarter = (i * 3) + 2, i
        data_monthly = data_monthly_total.iloc[idx_month:idx_month + seq_month]
        data_quarterly = data_quarterly_total.iloc[idx_quarter:idx_quarter + seq_quarter]
        data_monthly, data_quarterly = utils.set_index_to_date(data_monthly, data_quarterly)

        # normalize economic indicators
        if use_diff_norm == True:
            data_monthly = utils.normalize_for_dfm(data_monthly, trans_method)
            data_quarterly = utils.normalize_for_dfm(data_quarterly, trans_method)

        model = sm.tsa.DynamicFactorMQ(endog=data_monthly, endog_quarterly=data_quarterly, factors=group_info,
                                       factor_orders=1)
        results = model.fit(method='em')
        imputedX = results.fittedvalues.values

        if NCDE:
            imputedX = np.concatenate([dtime[idx_month:idx_month + seq_month], imputedX], axis=1)

        new_seq = np.expand_dims(imputedX, axis=0)

        if seq_data is None:
            seq_data = new_seq
        else:
            seq_data = np.concatenate([seq_data, new_seq], axis=0)

    # MinMaxScaling
    if use_min_max_norm == True:
        dtime = seq_data[:, :, 0:1]
        seq_wo_time = seq_data[:, :, 1:]
        normalized_wo_time = utils.normalize(seq_wo_time)
        seq_data = np.concatenate([dtime, normalized_wo_time], axis=2)

    total_num = seq_data.shape[0]
    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)

    if y_seq > 1:
        stacked_data_y = None
        for i in range(seq_quarter, seq_quarter + total_num):
            target_y = data_y[i:i + y_seq]
            target_y = np.expand_dims(target_y, axis=0)
            if stacked_data_y is None:
                stacked_data_y = target_y
            else:
                stacked_data_y = np.concatenate([stacked_data_y, target_y])

        trainX, trainy = seq_data[:train_len, :, :], stacked_data_y[:train_len, :, :]
        validX, validy = seq_data[train_len:valid_len, :, :], stacked_data_y[train_len:valid_len, :, :]
        testX, testy = seq_data[valid_len:, :, :], stacked_data_y[valid_len:, :, :]

    else:
        trainX, trainy = seq_data[:train_len, :, :], data_y[seq_quarter:train_len + seq_quarter]
        validX, validy = seq_data[train_len:valid_len, :, :], data_y[
                                                              train_len + seq_quarter:valid_len + seq_quarter]
        testX, testy = seq_data[valid_len:, :, :], data_y[valid_len + seq_quarter:]

    save_processed_data(folder_loc, train_seq, y_seq, NCDE, use_diff_norm, use_min_max_norm, train_split, valid_split,
                        test_split, trainX, trainy, validX, validy, testX, testy)

    return trainX, trainy, validX, validy, testX, testy


def with_factors(train_seq, y_seq, use_diff_norm=True, use_min_max_norm=True, train_split=0.7, valid_split=0.15,
                 test_split=0.15,
                 data='GDP_KOR',
                 target='GDP'):
    here = pathlib.Path(__file__).resolve().parent

    folder_loc = f'_processed_data/{data}/factors'
    checkFile_loc = f"{data}_trainX_{train_seq}_{y_seq}_True_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    if os.path.exists(here / folder_loc / checkFile_loc):
        return load_processed_data(folder_loc, train_seq, y_seq,
                                   True, use_diff_norm, use_min_max_norm,
                                   train_split, valid_split, test_split)

    # load monthly and quarterly data respectively
    df_data_monthly = pd.read_csv(here / f'{data}_monthly.csv')  # 28
    if 'miss' in data:
        data = re.sub('_miss_r[0-9]+', '', data)
    df_data_quarterly = pd.read_csv(here / f'{data}_quarterly.csv')  # 5

    df_data_monthly['Date'] = pd.to_datetime(df_data_monthly['Date'])
    df_data_quarterly['Date'] = pd.to_datetime(df_data_quarterly['Date'])

    seq_month, seq_quarter = train_seq, int(train_seq / 3)
    dtime = np.expand_dims(utils.time_normalize(df_data_monthly['Date']), axis=1)
    data_y = np.expand_dims(df_data_quarterly[target].values, axis=1)

    with open(here / f'{data}_info.json', 'r') as f:
        GDP_info = json.load(f)
        group_info = GDP_info['Group']
        trans_method = GDP_info['Trans']

    total_seq_data, total_seq_factors, total_seq_alpha = None, None, None
    time_step = range(len(df_data_quarterly) - seq_quarter - y_seq + 1)

    for i in tqdm(time_step):
        # slicing monthly and quarterly data by sequence length
        idx_month, idx_quarter = (i * 3) + 2, i
        data_X_monthly = df_data_monthly.iloc[idx_month:idx_month + seq_month]
        data_X_quarterly = df_data_quarterly.iloc[idx_quarter:idx_quarter + seq_quarter]

        # normalize economic indicators
        if use_diff_norm == True:
            data_X_quarterly = utils.normalize_for_dfm(data_X_quarterly, trans_method)
            data_X_monthly = utils.normalize_for_dfm(data_X_monthly, trans_method)

        # combine monthly and quarterly data
        data_X_quarterly = utils.convert_date_Q_to_M(data_X_quarterly)
        data_X = data_X_quarterly.merge(data_X_monthly, on='Date', how='right')
        data_X.set_index('Date', inplace=True)
        data_X.index = pd.DatetimeIndex(data_X.index).to_period('M')
        # print('##### data_X')
        # with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                        None):  # more options can be specified also
        #     print(data_X)

        # calculate factors at next sequence
        dfm_model = sm.tsa.DynamicFactorMQ(endog=data_X, k_endog_monthly=len(data_X_monthly.columns),
                                           factors=group_info, factor_orders=1)

        dfm_fitted = dfm_model.fit(method='em')
        dfm_forecast = dfm_fitted.forecast(steps=y_seq)
        dfm_fitted = dfm_fitted.apply(endog=dfm_forecast, k_endog_monthly=len(data_X_monthly.columns))

        if 'KOR' in data:
            dfm_states = dfm_fitted.states.smoothed
            # print('##### dfm_states')
            # with pd.option_context('display.max_rows', None, 'display.max_columns',
            #                        None):  # more options can be specified also
            #     print(dfm_states)
            # BUG FIX 2023.10.15 : factor extraction ['Global', 'Labor'] → ['Global', 'Real']
            dfm_factors = dfm_states[['Global', 'Real']].to_numpy()
        elif 'UK' in data:
            dfm_factors = dfm_fitted.factors.smoothed.values

        # for NCDE: augment time to data
        seq_data = np.concatenate([dtime[idx_month:idx_month + seq_month], data_X.values], axis=1)
        seq_data = np.expand_dims(seq_data, axis=0)

        # create sequence datasets
        if total_seq_data is None:
            total_seq_data = seq_data
        else:
            total_seq_data = np.concatenate([total_seq_data, seq_data], axis=0)

        if total_seq_factors is None:
            total_seq_factors = dfm_factors
        else:
            total_seq_factors = np.concatenate([total_seq_factors, dfm_factors], axis=0)

    # normalize : min-max scaling
    if use_min_max_norm == True:
        dtime = total_seq_data[:, :, 0:1]
        seq_wo_time = total_seq_data[:, :, 1:]
        normalized_wo_time = utils.normalize(seq_wo_time)
        total_seq_data = np.concatenate([dtime, normalized_wo_time], axis=2)
        total_seq_factors = utils.normalize(total_seq_factors)

    total_num = total_seq_data.shape[0]

    total_seq_y = None
    for i in range(seq_quarter, seq_quarter + total_num):
        seq_y = np.expand_dims(data_y[i:i + y_seq], axis=0)
        if total_seq_y is None:
            total_seq_y = seq_y
        else:
            total_seq_y = np.concatenate([total_seq_y, seq_y], axis=0)

    # split sequence datasets into train / valid / test datasets
    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
    trainX, trainy, trainF = total_seq_data[:train_len, :, :], total_seq_y[:train_len], total_seq_factors[:train_len]
    validX, validy, validF = total_seq_data[train_len:valid_len, :, :], total_seq_y[
                                                                        train_len:valid_len], total_seq_factors[
                                                                                              train_len:valid_len]
    testX, testy, testF = total_seq_data[valid_len:, :, :], total_seq_y[valid_len:], total_seq_factors[valid_len:]

    # save pre-processed data in npy
    save_processed_data(folder_loc, train_seq, y_seq,
                        True, use_diff_norm, use_min_max_norm,
                        train_split, valid_split, test_split,
                        trainX, trainy, validX, validy, testX, testy, trainF, validF, testF)

    return trainX, trainy, validX, validy, testX, testy, trainF, validF, testF


def FXR_with_factors(train_seq, y_seq, use_diff_norm=False, use_min_max_norm=True,
                     train_split=0.7, valid_split=0.15, test_split=0.15,
                     data='FXR_ETRI', target='원달러환율'):
    here = pathlib.Path(__file__).resolve().parent

    folder_loc = f'_processed_data/{data}/factors'
    checkFile_loc = f"{data}_trainX_{train_seq}_{y_seq}_True_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    if os.path.exists(here / folder_loc / checkFile_loc):
        return load_processed_data(folder_loc, train_seq, y_seq,
                                   True, use_diff_norm, use_min_max_norm,
                                   train_split, valid_split, test_split)

    df_data = pd.read_csv(here / f'{data}.csv')

    dtime = np.expand_dims(utils.time_normalize(df_data["Date"]), axis=1)
    df_data.set_index('Date', inplace=True)

    freq = 'D' if data == 'FXR_ETRI' else 'M'
    df_data.index = pd.DatetimeIndex(df_data.index).to_period(freq)
    data_y = np.expand_dims(df_data[target].values, axis=1) / 1000  # 단위: 천원

    with open(here / f'{data}_info.json', 'r') as f:
        FXR_info = json.load(f)
        group_info = FXR_info['그룹정보']
        trans_method = FXR_info['변환방법']

    timestep = range(len(df_data) - train_seq - y_seq + 1)
    total_seq_factors, total_seq_y = None, None
    for i in tqdm(timestep):
        data_X = df_data.iloc[i:i + train_seq]
        # data_X = df_data.iloc[0:i + train_seq]    # FIXME test accumulative method

        # normalize for dfm
        data_X = utils.normalize_for_dfm(data_X, trans_method)
        data_X['원달러환율'] = data_X['원달러환율'] / 1000

        # calculate factors at next time
        dfm_model = sm.tsa.DynamicFactorMQ(endog=data_X, factors=group_info, factor_orders=1)
        dfm_fitted = dfm_model.fit(method='em')
        dfm_forecast = dfm_fitted.forecast(steps=y_seq)
        dfm_fitted = dfm_fitted.extend(endog=dfm_forecast)
        # dfm_factors = dfm_fitted.states.smoothed.values
        dfm_states = dfm_fitted.states.smoothed

        # target factor : 전체(0), 환율(3)
        dfm_factors = dfm_states[['전체', '환율']].to_numpy()

        # for NCDE : augment time to data
        seq_data = np.concatenate([dtime[i:i + train_seq], data_X.to_numpy()], axis=1)
        seq_data = np.expand_dims(seq_data, axis=0)
        if total_seq_data is None:
            total_seq_data = seq_data
        else:
            total_seq_data = np.concatenate([total_seq_data, seq_data], axis=0)
        # print(f'seq_data shape:{seq_data.shape}')
        # print(f'total_seq_data shape:{total_seq_data.shape}')

        # factors
        # seq_factors = np.expand_dims(dfm_factors, axis=0)
        if total_seq_factors is None:
            total_seq_factors = dfm_factors
        else:
            total_seq_factors = np.concatenate([total_seq_factors, dfm_factors], axis=0)
        # print(f'seq_factors shape:{dfm_factors.shape}')
        # print(f'total_seq_factors shape:{total_seq_factors.shape}')

        # true y
        seq_y = np.expand_dims(data_y[i + train_seq:i + train_seq + y_seq], axis=0)
        if total_seq_y is None:
            total_seq_y = seq_y
        else:
            total_seq_y = np.concatenate([total_seq_y, seq_y], axis=0)

    # normalize : min-max scalling
    dtime = total_seq_data[:, :, 0:1]
    seq_wo_time = total_seq_data[:, :, 1:]
    normalized_wo_time = utils.normalize(seq_wo_time)
    total_seq_data = np.concatenate([dtime, normalized_wo_time], axis=2)

    total_num = total_seq_data.shape[0]

    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)

    trainX, trainy, trainF = total_seq_data[:train_len], total_seq_y[:train_len], total_seq_factors[:train_len]
    validX, validy, validF = total_seq_data[train_len:valid_len], total_seq_y[train_len:valid_len], \
        total_seq_factors[train_len:valid_len]
    testX, testy, testF = total_seq_data[valid_len:], total_seq_y[valid_len:], total_seq_factors[valid_len:]

    # save pre-processed data in npy
    save_processed_data(folder_loc, train_seq, y_seq,
                        True, use_diff_norm, use_min_max_norm,
                        train_split, valid_split, test_split,
                        trainX, trainy, validX, validy, testX, testy, trainF, validF, testF)

    return trainX, trainy, validX, validy, testX, testy, trainF, validF, testF


def for_DFM(train_seq, y_seq, train_split=0.7, valid_split=0.15, test_split=0.15, data='GDP_KOR', target='GDP'):
    here = pathlib.Path(__file__).resolve().parent

    df_data_monthly = pd.read_csv(here / f'{data}_monthly.csv')

    if 'miss' in data:
        data = re.sub('_miss_r[0-9]+', '', data)
    df_data_quarterly = pd.read_csv(here / f'{data}_quarterly.csv')

    df_data_monthly['Date'] = pd.to_datetime(df_data_monthly['Date'])
    df_data_quarterly['Date'] = pd.to_datetime(df_data_quarterly['Date'])
    data_y = np.expand_dims(df_data_quarterly[target].values, axis=1)

    df_data_quarterly = utils.convert_date_Q_to_M(df_data_quarterly)

    # group of economic indicators, this is used in DFM model for factors
    with open(here / f'{data}_info.json', 'r') as f:
        GDP_info = json.load(f)
        group_info = GDP_info['Group']
        trans_method = GDP_info['Trans']

    # normalize economic indicators
    df_data_monthly[df_data_monthly.columns.difference(['Date'])] = utils.normalize_for_dfm(
        df_data_monthly[df_data_monthly.columns.difference(['Date'])], trans_method)
    # print(df_data_monthly)
    df_data_quarterly[df_data_quarterly.columns.difference(['Date'])] = utils.normalize_for_dfm(
        df_data_quarterly[df_data_quarterly.columns.difference(['Date'])], trans_method)

    total_seq_data = []
    seq_month, seq_quarter = train_seq, int(train_seq / 3)
    time_step = range(len(df_data_quarterly) - seq_quarter - y_seq + 1)

    for i in tqdm(time_step):
        # slicing monthly and quarterly data by sequence length
        idx_month, idx_quarter = (i * 3) + 2, i
        data_X_monthly = df_data_monthly.iloc[idx_month:idx_month + seq_month]
        data_X_quarterly = df_data_quarterly.iloc[idx_quarter:idx_quarter + seq_quarter]

        # combine monthly and quarterly data
        data_X = data_X_quarterly.merge(data_X_monthly, on='Date', how='right')
        data_X.set_index('Date', inplace=True)
        data_X.index = pd.DatetimeIndex(data_X.index).to_period('M')

        total_seq_data.append(data_X)

    total_num = len(total_seq_data)

    total_seq_y = None
    for i in range(seq_quarter, seq_quarter + total_num):
        if total_seq_y is None:
            total_seq_y = np.expand_dims(data_y[i:i + y_seq], axis=0)
        else:
            total_seq_y = np.concatenate([total_seq_y, np.expand_dims(data_y[i:i + y_seq], axis=0)], axis=0)

    # split sequence datasets into train / valid / test datasets
    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
    trainX, trainy = total_seq_data[:train_len], total_seq_y[:train_len]
    validX, validy = total_seq_data[train_len:valid_len], total_seq_y[train_len:valid_len]
    testX, testy = total_seq_data[valid_len:], total_seq_y[valid_len:]

    # the number of monthly data, this is parameter of DFM model
    num_monthly = len(df_data_monthly.columns)

    return trainX, trainy, validX, validy, testX, testy, group_info, num_monthly

def FXR_for_DFM(train_seq, y_seq, freq, train_split=0.7, valid_split=0.15, test_split=0.15, data='FXR_ETRI', target='원달러환율'):
    here = pathlib.Path(__file__).resolve().parent

    df_data = pd.read_csv(here / f'{data}.csv')

    dtime = np.expand_dims(utils.time_normalize(df_data["Date"]), axis=1)
    df_data.set_index('Date', inplace=True)

    freq = 'D' if data == 'FXR_ETRI' else 'M'
    df_data.index = pd.DatetimeIndex(df_data.index).to_period(freq)
    data_y = np.expand_dims(df_data[target].values, axis=1) / 1000  # 단위: 천원

    with open(here / f'{data}_info.json', 'r') as f:
        FXR_info = json.load(f)
        group_info = FXR_info['그룹정보']
        trans_method = FXR_info['변환방법']

    # normalize for dfm
    df_data = utils.normalize_for_dfm(df_data, trans_method)
    # FIXME utils 내 반영
    df_data[target] = df_data[target] / 1000

    timestep = range(len(df_data) - train_seq - y_seq + 1)
    total_seq_data = []
    total_seq_y = None
    for i in tqdm(timestep):
        seq_data = df_data.iloc[i:i + train_seq]
        total_seq_data.append(seq_data)

        seq_y = np.expand_dims(data_y[i + train_seq:i + train_seq + y_seq], axis=0)
        if total_seq_y is None:
            total_seq_y = seq_y
        else:
            total_seq_y = np.concatenate([total_seq_y, seq_y], axis=0)

    total_num = len(total_seq_data)

    # split sequence datasets into train / valid / test datasets
    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
    trainX, trainy = total_seq_data[:train_len], total_seq_y[:train_len]
    validX, validy = total_seq_data[train_len:valid_len], total_seq_y[train_len:valid_len]
    testX, testy = total_seq_data[valid_len:], total_seq_y[valid_len:]

    return trainX, trainy, validX, validy, testX, testy, group_info, None



class NCDE_USE_TYPE(Enum):
    NONE = 1  # not use
    ONLY = 2  # only use (coeffs)
    WITH = 3  # use with


def create_loaders(trainX, trainy, validX, validy, testX, testy, ncde_use_type, device, batch_size=128, trainF=None,
                   validF=None, testF=None):
    if ncde_use_type == NCDE_USE_TYPE.ONLY:
        # to tensor and device
        trainX, trainy, validX, validy, testX, testy = to_tensor_device(device, trainX, trainy, validX, validy, testX,
                                                                        testy)

        train_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(trainX)
        valid_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(validX)
        test_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(testX)

        train_dataset = TensorDataset(train_coeff, trainy)
        valid_dataset = TensorDataset(valid_coeff, validy)
        test_dataset = TensorDataset(test_coeff, testy)

    elif ncde_use_type == NCDE_USE_TYPE.WITH:

        # to tensor and device
        trainX, trainF, trainy, validX, validF, validy, testX, testF, testy \
            = to_tensor_device(device, trainX, trainF, trainy, validX, validF, validy, testX, testF, testy)

        train_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(trainX)
        valid_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(validX)
        test_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(testX)

        train_dataset = TensorDataset(trainX, train_coeff, trainF, trainy)
        valid_dataset = TensorDataset(validX, valid_coeff, validF, validy)
        test_dataset = TensorDataset(testX, test_coeff, testF, testy)

    elif ncde_use_type == NCDE_USE_TYPE.NONE:
        # to tensor and device
        trainX, trainy, validX, validy, testX, testy = to_tensor_device(device, trainX, trainy, validX, validy, testX,
                                                                        testy)
        train_dataset = TensorDataset(trainX, trainy)
        valid_dataset = TensorDataset(validX, validy)
        test_dataset = TensorDataset(testX, testy)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader


def to_tensor_device(device, *dataset):
    return tuple(torch.Tensor(data).to(device) for data in dataset)


def load_processed_data(data_path, train_seq, y_seq,
                        NCDE, use_diff_norm, use_min_max_norm,
                        train_split, valid_split, test_split):
    here = pathlib.Path(__file__).resolve().parent

    # for example, GDP, ER, and so on.
    dataset_name = data_path.split('/')[1]

    trainX_file = f"{dataset_name}_trainX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    trainy_file = f"{dataset_name}_trainy_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    validX_file = f"{dataset_name}_validX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    validy_file = f"{dataset_name}_validy_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    testX_file = f"{dataset_name}_testX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    testy_file = f"{dataset_name}_testy_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    trainX, trainy = np.load(here / data_path / trainX_file), np.load(here / data_path / trainy_file)
    validX, validy = np.load(here / data_path / validX_file), np.load(here / data_path / validy_file)
    testX, testy = np.load(here / data_path / testX_file), np.load(here / data_path / testy_file)

    # check factors/alpha info (used in related work for DFM)
    trainF_file = f"{dataset_name}_trainF_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    validF_file = f"{dataset_name}_validF_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    testF_file = f"{dataset_name}_testF_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    if os.path.exists(here / data_path / trainF_file) \
            and os.path.exists(here / data_path / validF_file) \
            and os.path.exists(here / data_path / testF_file):
        trainF, validF, testF = np.load(here / data_path / trainF_file), \
            np.load(here / data_path / validF_file), \
            np.load(here / data_path / testF_file)
        return trainX, trainy, validX, validy, testX, testy, trainF, validF, testF

    return trainX, trainy, validX, validy, testX, testy


def save_processed_data(data_path, train_seq, y_seq,
                        NCDE, use_diff_norm, use_min_max_norm,
                        train_split, valid_split, test_split,
                        trainX, trainy, validX, validy, testX, testy,
                        trainF=None, validF=None, testF=None,
                        trainA=None, validA=None, testA=None):
    here = pathlib.Path(__file__).resolve().parent

    if not os.path.exists(here / data_path):
        os.makedirs(here / data_path)

    # for example, GDP, ER, and so on.
    dataset_name = data_path.split('/')[1]

    trainX_file = f"{dataset_name}_trainX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    trainy_file = f"{dataset_name}_trainy_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    validX_file = f"{dataset_name}_validX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    validy_file = f"{dataset_name}_validy_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    testX_file = f"{dataset_name}_testX_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
    testy_file = f"{dataset_name}_testy_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"

    np.save(f"{here}/{data_path}/{trainX_file}", trainX)
    np.save(f"{here}/{data_path}/{trainy_file}", trainy)
    np.save(f"{here}/{data_path}/{validX_file}", validX)
    np.save(f"{here}/{data_path}/{validy_file}", validy)
    np.save(f"{here}/{data_path}/{testX_file}", testX)
    np.save(f"{here}/{data_path}/{testy_file}", testy)

    if trainF is not None:
        trainF_loc = f"{dataset_name}_trainF_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
        np.save(f"{here}/{data_path}/{trainF_loc}", trainF)

    if validF is not None:
        validF_loc = f"{dataset_name}_validF_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
        np.save(f"{here}/{data_path}/{validF_loc}", validF)

    if testF is not None:
        testF_loc = f"{dataset_name}_testF_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
        np.save(f"{here}/{data_path}/{testF_loc}", testF)

    if trainA is not None:
        trainA_loc = f"{dataset_name}_trainA_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
        np.save(f"{here}/{data_path}/{trainA_loc}", trainA)

    if validA is not None:
        validA_loc = f"{dataset_name}_validA_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
        np.save(f"{here}/{data_path}/{validA_loc}", validA)

    if testA is not None:
        testA_loc = f"{dataset_name}_testA_{train_seq}_{y_seq}_{NCDE}_{use_diff_norm}_{use_min_max_norm}_{train_split}_{valid_split}_{test_split}.npy"
        np.save(f"{here}/{data_path}/{testA_loc}", testA)

    return None
