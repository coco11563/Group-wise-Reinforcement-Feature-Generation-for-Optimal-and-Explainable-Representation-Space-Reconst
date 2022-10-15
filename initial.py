import os
import sys

import torch
import warnings


BASE_DIR = os.path.dirname(os.path.abspath('.'))

sys.path.append(BASE_DIR)

from utils.logger import *

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

os.environ['DATA_ABS_PATH'] = BASE_DIR + '/data/processed'

import argparse


def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--name', type=str, default='wine_red',
                        help='data name')
    parser.add_argument('--log-level', type=str, default='info', help=
    'log level, check the utils.logger')
    parser.add_argument('--episodes', type=int, default=10, help=
    'episodes for training')
    parser.add_argument('--steps', type=int, default=10, help=
    'steps for each episode')
    parser.add_argument('--enlarge_num', type=int, default=4, help=
    'feature space enlarge')
    parser.add_argument('--memory', type=int, default=8, help='memory capacity')
    parser.add_argument('--a', type=float, default=1, help='a')
    parser.add_argument('--b', type=float, default=1, help='b')
    parser.add_argument('--c', type=float, default=1, help='c')
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--replay-strategy', type=str, default='random')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--init-w', type=float, default=1e-1)
    parser.add_argument('--id', type=str, default='0', help='give this exp a special id!')
    # -c removing the feature clustering step of GRFG
    # -d using euclidean distance as feature distance metric in the M-clustering of GRFG
    # -b -u Third, we developed GRFG−𝑢 and GRFG−𝑏 by using random in the two feature generation scenarios
    parser.add_argument('--ablation-mode', type=str, default='')
    parser.add_argument('--out-put', type=str, default='.')

    # priority experiment replay related parameter
    # parser.add_argument('--per-alpha', type=float, default=0.7)
    # parser.add_argument('--per-beta-zero', type=float, default=0.5)
    # parser.add_argument('--per-learn-start', type=int, default=1000)
    # parser.add_argument('--per-steps', type=int, default=100000)
    # parser.add_argument('--per-partition-num', type=int, default=100)
    # parser.add_argument('--per-batch-size', type=int, default=32)
    # parser.add_argument('--per-replace-old', type=bool, default=True)
    # parser.add_argument('--per-priority-size', type=int, default=100)

    args, _ = parser.parse_known_args()
    return args
