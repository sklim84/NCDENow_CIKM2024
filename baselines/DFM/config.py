import argparse

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--model', type=str, default='dfm')
parser.add_argument('--n_factors', type=int, default=2)
parser.add_argument('--factor_orders', type=int, default=1)
parser.add_argument('--train_seq', type=int, default=15)
parser.add_argument('--y_seq', type=int, default=1)
# Dataset: GDP_KOR, GRP_UK, GDP_KOR_miss_r10, GDP_KOR_miss_r20, GDP_UK_miss_r10, GDP_UK_miss_r20
parser.add_argument('--data', type=str, default='GDP_KOR')

# etc
parser.add_argument('--log_path', type=str, default='../../_logs/')
parser.add_argument('--results_path', type=str, default='../../_results')
parser.add_argument('--save_values', action='store_true', help='', default=False)

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
