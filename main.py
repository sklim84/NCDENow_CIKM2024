import os
import sys
import warnings
from datetime import datetime

import math
import torch
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import socket

from _commons.pytorchtools import EarlyStopping
from _commons.utils import save_metrics, exists_metrics, get_logger, fix_randomness, save_values, count_parameters
from _datasets import preprocessor
from _datasets.preprocessor import create_loaders, NCDE_USE_TYPE
from config import get_config
from model import NCDENow

warnings.filterwarnings('ignore')


def training(model, dataloader, optimizer, device):
    true_ys, pred_ys = torch.Tensor().to(device), torch.Tensor().to(device)
    mse_loss_y = torch.nn.MSELoss(reduction='mean')

    for i, batch in enumerate(dataloader):
        batch_x, batch_coeffs, batch_f, batch_y = batch

        batch_y = batch_y.squeeze(-1)

        optimizer.zero_grad()
        pred_y, _, _ = model(batch_x, batch_coeffs, batch_f)

        true_ys = torch.concat([true_ys, batch_y])
        pred_ys = torch.concat([pred_ys, pred_y])

        loss = mse_loss_y(pred_y, batch_y)
        loss.backward()
        optimizer.step()

    mse_value_y = mse_loss_y(pred_ys, true_ys)
    mape_value_y = mean_absolute_percentage_error(true_ys.detach().cpu().numpy(), pred_ys.detach().cpu().numpy()) * 100
    return mse_value_y, mape_value_y


def evaluating(model, dataloader, device):
    with torch.no_grad():
        true_ys, pred_ys = torch.Tensor().to(device), torch.Tensor().to(device)
        pred_as, pred_bs = torch.Tensor().to(device), torch.Tensor().to(device)
        pred_fs = torch.Tensor().to(device)

        for i, batch in enumerate(dataloader):
            batch_x, batch_coeffs, batch_f, batch_y = batch
            batch_y = batch_y.squeeze(-1)

            pred_y, pred_a, pred_b = model(batch_x, batch_coeffs, batch_f)
            true_ys = torch.concat([true_ys, batch_y])
            pred_ys = torch.concat([pred_ys, pred_y])
            pred_as = torch.concat([pred_as, pred_a])
            pred_bs = torch.concat([pred_bs, pred_b])
            pred_fs = torch.concat([pred_fs, batch_f])
            print(f'batch_f shape: {batch_f.shape}')
            print(f'pred_fs shape: {pred_fs.shape}')

        mse_loss = torch.nn.MSELoss(reduction='mean')
        mse_value = mse_loss(pred_ys, true_ys)
        true_ys, pred_ys = true_ys.detach().cpu().numpy(), pred_ys.detach().cpu().numpy()
        pred_as, pred_bs = pred_as.detach().cpu().numpy(), pred_bs.detach().cpu().numpy()
        pred_fs = pred_fs.detach().cpu().numpy()
        mape_value = mean_absolute_percentage_error(true_ys, pred_ys) * 100

    return mse_value, mape_value, true_ys, pred_ys, pred_as, pred_bs, pred_fs


if __name__ == "__main__":
    # set configuration
    args = get_config()

    # get logger
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    print(args.log_path)
    logger = get_logger(args.log_path, f'{args.model}_{datetime.now().strftime("%Y%m%d")}.log')
    logger.info(args)
    print(args)

    # check existence of experiments results
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Server IP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('www.google.com', 443))
    server_ip = sock.getsockname()[0]
    server_ip = server_ip.split('.')[3]

    metric_save_path = os.path.abspath(os.path.join(args.results_path, f'{args.data}_{args.model}_metrics_{server_ip}.csv'))
    arg_names = ['model', 'cde_type', 'data', 'train_seq', 'y_seq',
                 'epochs', 'lr', 'batch', 'weight_decay', 'hidden_size', 'hidden_hidden_size', 'n_layers', 'ode_method',
                 'early_stopping_patience', 'use_diff_norm', 'use_min_max_norm', 'seed']
    if exists_metrics(metric_save_path, args, arg_names):
        logger.info(f'There exist experiments results! - {args}')
        sys.exit()

    # fix randomness
    fix_randomness(args.seed)
    device = torch.device(f'cuda:{args.device}')
    epochs = args.epochs

    if 'KOR' in args.data:
        target = 'GDP'
    elif 'UK' in args.data:
        target = 'GDPP'
    else:
        target = ''

    trainX, trainy, validX, validy, testX, testy, trainF, validF, testF \
        = preprocessor.with_factors(args.train_seq, args.y_seq,
                                    use_diff_norm=args.use_diff_norm,
                                    use_min_max_norm=args.use_min_max_norm,
                                    data=args.data,
                                    target=target)
    train_dataloader, valid_dataloader, test_dataloader = create_loaders(
        trainX, trainy, validX, validy, testX, testy, NCDE_USE_TYPE.WITH, device,
        args.batch, trainF, validF, testF)

    print(f'##### train dataset: {len(train_dataloader.dataset)}')
    print(f'##### valid dataset: {len(valid_dataloader.dataset)}')
    print(f'##### test dataset: {len(test_dataloader.dataset)}')

    input_size, output_size = trainX.shape[2], args.y_seq
    print(f'trainX shape: {trainX.shape}')
    model = NCDENow(args.fe_type, args.cde_type, input_size, args.hidden_size, args.hidden_hidden_size,
                    output_size, args.n_factors, args.n_layers, args.ode_method, device)
    model.to(device)

    # print model information
    logger.info(model)
    total_params, trainable_params = count_parameters(model)
    logger.info(f'##### total model parameters: {total_params}')
    logger.info(f'##### trainable model parameters: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    best_epoch, best_train_mse, best_valid_mse, best_test_mse = 0, math.inf, math.inf, math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        train_mse_loss, train_mape_loss = training(model, train_dataloader, optimizer, device)

        model.eval()
        valid_mse_loss, valid_mape_loss, _, _, _, _, _ = evaluating(model, valid_dataloader, device)
        test_mse_loss, test_mape_loss, true_ys, pred_ys, pred_as, pred_bs, pred_fs = evaluating(model, test_dataloader, device)

        logger.info(f'{"[○ PROC STEP MSE]":{20}}\tEpoch:\t{epoch:{4}}\t'
                    f'Train:\t{train_mse_loss.item():{10}.4f}\t'
                    f'Valid:\t{valid_mse_loss.item():{10}.4f}\t'
                    f'Test:\t{test_mse_loss.item():{10}.4f}')
        logger.info(f'{"[○ PROC STEP MAPE]":{20}}\tEpoch:\t{epoch:{4}}\t'
                    f'Train:\t{train_mape_loss.item():{10}.4f}\t'
                    f'Valid:\t{valid_mape_loss.item():{10}.4f}\t'
                    f'Test:\t{test_mape_loss.item():{10}.4f}')

        if best_valid_mse > valid_mse_loss:
            best_epoch, best_train_mse, best_valid_mse, best_test_mse, best_train_mape, best_valid_mape, best_test_mape = \
                epoch, train_mse_loss.item(), valid_mse_loss.item(), test_mse_loss.item(), train_mape_loss, valid_mape_loss, test_mape_loss

            logger.info(f'{"[● UPDT BEST MSE]":{20}}\tEpoch:\t{best_epoch:{4}}\t'
                        f'Train:\t{best_train_mse:{10}.4f}\t'
                        f'Valid:\t{best_valid_mse:{10}.4f}\t'
                        f'Test:\t{best_test_mse:{10}.4f}')
            logger.info(f'{"[● UPDT BEST MAPE]":{20}}\tEpoch:\t{best_epoch:{4}}\t'
                        f'Train:\t{best_train_mape:{10}.4f}\t'
                        f'Valid:\t{best_valid_mape:{10}.4f}\t'
                        f'Test:\t{best_test_mape:{10}.4f}')

            if args.save_values:
                print(save)
                values_save_path = os.path.abspath(os.path.join(args.results_path, 'values'))
                model_name = '_'.join([args.model, args.cde_type])
                save_values(values_save_path, args.data, model_name, args.train_seq, args.y_seq, true_ys, pred_ys)
                # save alpha and beta
                np.save(os.path.join(values_save_path, f'{args.data}_{model_name}_best_test_preda_{args.train_seq}_{args.y_seq}.npy'), pred_as)
                np.save(os.path.join(values_save_path, f'{args.data}_{model_name}_best_test_predb_{args.train_seq}_{args.y_seq}.npy'), pred_bs)
                np.save(os.path.join(values_save_path, f'{args.data}_{model_name}_best_test_predf_{args.train_seq}_{args.y_seq}.npy'), pred_fs)

        # check early stopping
        if args.early_stopping_patience > 0:
            early_stopping(valid_mse_loss.item(), model)
            if early_stopping.early_stop:
                logger.info('# early stopping!')
                break

    logger.info(f'{"[★ FINL BEST MSE]":{20}}\tEpoch:\t{best_epoch:{4}}\t'
                f'Train:\t{best_train_mse:{10}.4f}\t'
                f'Valid:\t{best_valid_mse:{10}.4f}\t'
                f'Test:\t{best_test_mse:{10}.4f}')
    logger.info(f'{"[★ FINL BEST MAPE]":{20}}\tEpoch:\t{best_epoch:{4}}\t'
                f'Train:\t{best_train_mape:{10}.4f}\t'
                f'Valid:\t{best_valid_mape:{10}.4f}\t'
                f'Test:\t{best_test_mape:{10}.4f}')

    # save_results
    metric_names = ['mse', 'mape']
    metrics = [best_test_mse, best_test_mape]
    vars(args)['best_epoch'] = best_epoch
    arg_names.insert(arg_names.index('epochs') + 1, 'best_epoch')
    save_metrics(metric_save_path, args, arg_names, metrics, metric_names)

    print(f'##### best MSE: {best_test_mse}, MAPE: {best_test_mape}')
