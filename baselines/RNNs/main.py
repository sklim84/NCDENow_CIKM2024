import os
import sys

parent_path = (os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(parent_path)
parent_parent_path = (os.path.dirname(os.path.abspath(parent_path)))
sys.path.append(parent_parent_path)

import socket

import torch
from _datasets import preprocessor

from config import get_config
import models

from _datasets.preprocessor import create_loaders, NCDE_USE_TYPE
import math
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
from _commons.pytorchtools import EarlyStopping
from _commons.utils import save_metrics, exists_metrics, get_logger, fix_randomness, save_values, count_parameters


def create_model(args):
    model_name, data = args.model, args.data

    if model_name == 'ncde_naive':
        data_preprocessor = preprocessor.default
    else:
        data_preprocessor = preprocessor.with_impute

    if 'ncde' in model_name:
        isNCDE, typeNCDE = True, NCDE_USE_TYPE.ONLY
        model = models.NeuralCDE
    else:
        isNCDE, typeNCDE = False, NCDE_USE_TYPE.NONE
        if model_name == 'mlp':
            model = models.MLPNet
        elif model_name == 'rnn':
            model = models.RNNNet
        elif model_name == 'lstm':
            model = models.LSTMNet
        elif model_name == 'gru':
            model = models.GRUNet

    return model, data_preprocessor, isNCDE, typeNCDE


def training(model, dataloader, optimizer, device):
    true_ys, pred_ys = torch.Tensor().to(device), torch.Tensor().to(device)
    mse_loss = torch.nn.MSELoss(reduction='mean')
    for batch in dataloader:
        batch_coeffs, batch_y = batch
        batch_y = batch_y.squeeze(-1)

        optimizer.zero_grad()
        pred_y = model(batch_coeffs)
        true_ys = torch.concat([true_ys, batch_y])
        pred_ys = torch.concat([pred_ys, pred_y])
        loss = mse_loss(pred_y, batch_y)

        loss.backward()
        optimizer.step()

    mseLoss = mse_loss(pred_ys, true_ys)
    mapeLoss = mean_absolute_percentage_error(true_ys.detach().cpu().numpy(), pred_ys.detach().cpu().numpy()) * 100

    return mseLoss, mapeLoss


def evaluating(model, dataloader, device):
    with torch.no_grad():
        true_ys, pred_ys = torch.Tensor().to(device), torch.Tensor().to(device)

        for batch in dataloader:
            batch_coeffs, batch_y = batch
            batch_y = batch_y.squeeze(-1)

            pred_y = model(batch_coeffs)
            true_ys = torch.concat([true_ys, batch_y])
            pred_ys = torch.concat([pred_ys, pred_y])

        mse_loss = torch.nn.MSELoss(reduction='mean')
        mseLoss = mse_loss(pred_ys, true_ys)
        true_ys, pred_ys = true_ys.detach().cpu().numpy(), pred_ys.detach().cpu().numpy()
        mapeLoss = mean_absolute_percentage_error(true_ys, pred_ys) * 100

    return mseLoss, mapeLoss, true_ys, pred_ys


if __name__ == "__main__":

    # set configuration
    args = get_config()

    # get logger
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logger = get_logger(args.log_path, f'{args.model}_{datetime.now().strftime("%Y%m%d")}.log')
    logger.info(args)
    print(args)

    # check existence of experiments results
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Server IP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('www.naver.com', 443))
    server_ip = sock.getsockname()[0]
    server_ip = server_ip.split('.')[3]

    metric_save_path = os.path.abspath(os.path.join(args.results_path, f'{args.data}_{args.model}_metrics_{server_ip}.csv'))
    arg_names = ['model', 'data', 'train_seq', 'y_seq',
                 'epochs', 'lr', 'batch', 'weight_decay', 'hidden_size', 'hidden_hidden_size', 'n_layers',
                 'early_stopping_patience', 'use_diff_norm', 'use_min_max_norm', 'seed']
    if exists_metrics(metric_save_path, args, arg_names):
        logger.info(f'There exist experiments results! - {args}')
        sys.exit()

    # fix randomness
    fix_randomness(args.seed)
    # GPU setting
    device = torch.device(f'cuda:{args.device}')
    # get the model and the pre-processing ftn
    target_model, data_preprocessor, isNCDE, typeNCDE = create_model(args)

    if 'KOR' in args.data:
        target = 'GDP'
    elif 'UK' in args.data:
        target = 'GDPP'
    else:
        target = ''
    trainX, trainy, validX, validy, testX, testy = data_preprocessor(train_seq=args.train_seq, y_seq=args.y_seq,
                                                                     NCDE=isNCDE,
                                                                     use_diff_norm=args.use_diff_norm,
                                                                     use_min_max_norm=args.use_min_max_norm,
                                                                     data=args.data, target=target)

    train_dataloader, valid_dataloader, test_dataloader = create_loaders(
        trainX, trainy, validX, validy, testX, testy, ncde_use_type=typeNCDE, device=device, batch_size=args.batch)

    input_size, output_size = trainX.shape[2], args.y_seq
    model = target_model(input_size, args.hidden_size, output_size, args.n_layers, args.hidden_hidden_size)
    model.to(device)

    logger.info(model)
    total_params, trainable_params = count_parameters(model)
    logger.info(f'##### total model parameters: {total_params}')
    logger.info(f'##### trainable model parameters: {trainable_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    best_epoch, best_train_mse, best_valid_mse, best_test_mse = 0, math.inf, math.inf, math.inf

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse_loss, train_mape_loss = training(model, train_dataloader, optimizer, device)

        model.eval()
        valid_mse_loss, valid_mape_loss, _, _ = evaluating(model, valid_dataloader, device)
        test_mse_loss, test_mape_loss, true_ys, pred_ys = evaluating(model, test_dataloader, device)

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
                values_save_path = os.path.abspath(os.path.join(args.results_path, 'values'))
                save_values(values_save_path, args.data, args.model, args.train_seq, args.y_seq, true_ys, pred_ys)

        # check early stopping
        if args.early_stopping_patience > 0:
            early_stopping(valid_mse_loss, model)
            if early_stopping.early_stop:
                logger.info('early stopping!')
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
