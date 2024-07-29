import argparse

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--model', type=str, default='ncdenow')
parser.add_argument('--seed', type=int, default=0)
# Dataset: GDP_KOR, GDP_UK, GDP_KOR_miss_r10, GDP_KOR_miss_r20, GDP_UK_miss_r10, GDP_UK_miss_r20
parser.add_argument('--data', type=str, default='GDP_KOR')
parser.add_argument('--train_seq', type=int, default=15)
parser.add_argument('--y_seq', type=int, default=1)

# Hyperparameter
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--early_stopping_patience', type=int, default=5)
parser.add_argument('--n_factors', type=int, default=2)  # GDP_KOR (2), GDP_UK (8)
parser.add_argument('--factor_orders', type=int, default=1)
parser.add_argument('--cde_type', type=str, default='MLPCDEFunc',
                    help='\'MLPCDEFunc\' or \'GRUCDEFunc\'')
parser.add_argument('--fe_type', type=str, default='FactorAnalysisEncoder',
                    help='\'FactorAnalysisEncoder\' or \'GRUEncoder\'')
parser.add_argument('--hidden_size', type=int, default=16)
parser.add_argument('--hidden_hidden_size', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--ode_method', type=str, default='rk4')
parser.add_argument('--use_diff_norm', action='store_true', help='', default=True)
parser.add_argument('--use_min_max_norm', action='store_true', help='', default=True)

# Etc
parser.add_argument('--log_path', type=str, default='./_logs/')
parser.add_argument('--results_path', type=str, default='./_results/')
parser.add_argument('--save_values', action='store_true', help='', default=True)


def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
