import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
import sys


from feature_env import FeatureEnv, REPLAY
from initial import init_param
from model import operation_set, O1, O2, OpDQNNetwork, ClusterDQNNetwork
from replay import RandomClusterReplay, RandomOperationReplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
base_path = BASE_DIR + '/data/processed'
import warnings
import torch
import pandas as pd
import numpy as np

from utils.logger import *

import warnings

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


def train(param):
    cuda_info = None
    info(f'running experiment on cpu')
    NAME = param['name']
    ENV = FeatureEnv(task_name=NAME, ablation_mode=param['ablation_mode'])
    data_path = os.path.join(base_path, NAME + '.hdf')
    info('read the data from {}'.format(data_path))
    SAMPLINE_METHOD = param['replay_strategy']
    assert SAMPLINE_METHOD in REPLAY
    D_OPT_PATH = './tmp/' + NAME + '_' + \
                 SAMPLINE_METHOD + '_' + '/'
    info('opt path is {}'.format(D_OPT_PATH))
    Dg = pd.read_hdf(data_path)
    feature_names = list(Dg.columns)
    info('initialize the features...')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = Dg.values[:, :-1]
    X = scaler.fit_transform(X)
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    print(feature_names)
    Dg.columns = [str(i) for i in feature_names]
    D_OPT = Dg.copy()
    hidden = param['hidden_size']

    OP_DIM = len(operation_set)
    STATE_DIM = 0
    STATE_DIM += hidden
    mem_1_dim = STATE_DIM
    mem_op_dim = STATE_DIM
    info(f'initial memories with {SAMPLINE_METHOD}')
    BATCH_SIZE = param['batch_size']
    MEMORY_CAPACITY = param['memory']
    ENV.report_performance(Dg, D_OPT)
    if SAMPLINE_METHOD == 'random':
        cluster1_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, cuda_info)
        cluster2_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, cuda_info, OP_DIM)
        op_mem = RandomOperationReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_op_dim, cuda_info)
    else:
        error(f'unsupported sampling method {SAMPLINE_METHOD}')
        assert False
    ENT_WEIGHT = param['ent_weight']
    LR = 0.01
    init_w = param['init_w']
    model_cluster1 = ClusterDQNNetwork(state_dim=STATE_DIM, cluster_state_dim=STATE_DIM, hidden_dim=STATE_DIM * 2,
                                        memory=cluster1_mem,
                                        ent_weight=ENT_WEIGHT, select='head',
                                        gamma=0.99,
                                        device=cuda_info, init_w=init_w)

    model_cluster2 = ClusterDQNNetwork(state_dim=STATE_DIM + OP_DIM, cluster_state_dim=STATE_DIM, hidden_dim=(STATE_DIM + OP_DIM) * 2,
                                        memory=cluster2_mem,
                                        ent_weight=ENT_WEIGHT, select='tail',
                                        gamma=0.99,
                                        device=cuda_info, init_w=init_w)

    model_op = OpDQNNetwork(state_dim=STATE_DIM, cluster_state_dim=STATE_DIM, hidden_dim=STATE_DIM * 2,
                             memory=op_mem, ent_weight=ENT_WEIGHT, gamma=0.99, device=cuda_info, init_w=init_w)
    if cuda_info:
        model_cluster1 = model_cluster1.cuda()
        model_cluster2 = model_cluster2.cuda()
        model_op = model_op.cuda()
    optimizer_op = torch.optim.Adam(model_op.parameters(), lr=LR)
    optimizer_c2 = torch.optim.Adam(model_cluster2.parameters(), lr=LR)
    optimizer_c1 = torch.optim.Adam(model_cluster1.parameters(), lr=LR)

    EPISODES = param['episodes']
    STEPS = param['steps']
    episode = 0
    old_per = ENV.get_reward(Dg)
    best_per = old_per
    base_per = old_per
    info(f'start training, the original performance is {old_per}')
    D_original = Dg.copy()
    steps_done = 0
    FEATURE_LIMIT = Dg.shape[1] * param['enlarge_num']
    best_step = -1
    best_episode = -1
    training_start_time = time.time()
    while episode < EPISODES:
        step = 0
        Dg = D_original.copy()
        best_per_opt = []
        while step < STEPS:
            steps_done += 1
            step_start_time = time.time()
            clusters = ENV.cluster_build(Dg.values[:, :-1], Dg.values[:, -1], cluster_num=3)
            info(f'current cluster : {clusters}')
            acts1, action_emb, f_names1, f_cluster1, action_list, state_emb = \
                model_cluster1.select_action(clusters=clusters, X=Dg.values[:, :-1], feature_names=feature_names, steps_done=steps_done)
            op, op_name = model_op.select_operation(action_emb, steps_done=steps_done)

            if op_name in O1:
                Dg, is_op = model_cluster1.op(Dg, f_cluster1, f_names1, op_name)
                if not is_op:
                    continue
            else:
                acts2, action_emb2, f_names2, f_cluster2, _, state_emb2 = \
                    model_cluster2.select_action(clusters, Dg.values[:, :-1], feature_names,
                                                 op_name, cached_state_embed=state_emb, cached_cluster_state=action_list, steps_done=steps_done)
                if FEATURE_LIMIT * 4 < (f_cluster1.shape[1] * f_cluster2.shape[1]):
                    continue
                Dg, is_op = model_cluster1.op(Dg, f_cluster1, f_names1, op_name, f_cluster2, f_names2)
                if not is_op:
                    continue
            feature_names = list(Dg.columns)
            new_per = ENV.get_reward(Dg)
            reward = new_per - old_per
            r_c1, r_op, r_c2 = param['a'] * reward, param['b'] * reward, param['c'] * reward
            if new_per > best_per:
                best_step = step
                best_episode = episode
                best_per = new_per
                D_OPT = Dg.copy()
            old_per = new_per
            clusters_ = ENV.cluster_build(Dg.values[:, :-1], Dg.values[:, -1], cluster_num=3)
            acts_, action_emb_, f_names1_, f_cluster1_, action_list_, state_emb_ = \
                model_cluster1.select_action(clusters_, Dg.values[:, :-1], feature_names, for_next=True)
            op_, op_name_ = model_op.select_operation(action_emb_, for_next=True)
            if op_name in O2:
                _, action_emb2_, _, _, _, state_emb2_ = \
                    model_cluster2.select_action(clusters_, Dg.values[:, :-1], feature_names,
                                                 op=op_name_, cached_state_embed=state_emb_,
                                                 cached_cluster_state=action_list_, for_next=True)
                model_cluster2.store_transition(state_emb2, action_emb2, r_c2, state_emb2_, action_emb2_) #s1, a1, r, s2, a2
            model_cluster1.store_transition(state_emb, action_emb, r_c1, state_emb_, action_emb_)
            model_op.store_transition(action_emb, op, r_op, action_emb_)
            if model_cluster1.memory.memory_counter >= model_cluster1.memory.MEMORY_CAPACITY:
                info('start to learn in model_c1')
                model_cluster1.learn(optimizer_c1)
            if model_cluster2.memory.memory_counter >= model_cluster2.memory.MEMORY_CAPACITY:
                info('start to learn in model_c2')
                model_cluster2.learn(optimizer_c2)
            if model_op.memory.memory_counter >= model_op.memory.MEMORY_CAPACITY:
                info('start to learn in model_op')
                model_op.learn(optimizer_op)
            if Dg.shape[1] > FEATURE_LIMIT:
                selector = SelectKBest(mutual_info_regression, k=FEATURE_LIMIT)\
                    .fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
                cols = selector.get_support()
                X_new = Dg.iloc[:, :-1].loc[:, cols]
                Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
            info(
                'New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}'
                .format(new_per, best_per, best_episode, best_step, base_per))
            info('Episode {}, Step {} ends!'.format(episode, step))
            best_per_opt.append(best_per)
            info('Current spend time for step-{} is: {:.1f}s'.format(step,
                                                                     time.time() - step_start_time))
            step += 1
        if episode % 5 == 0:
            info('Best performance is: {:.6f}'.format(np.max(best_per_opt)))
            info('Episode {} ends!'.format(episode))
        episode += 1
    info('Total training time for is: {:.1f}s'.format(time.time() -
                                                      training_start_time))
    info('Exploration ends!')
    info('Begin evaluation...')
    final = ENV.report_performance(D_original, D_OPT)
    info('Total using time: {:.1f}s'.format(time.time() - training_start_time))
    if not os.path.exists(D_OPT_PATH):
        os.mkdir(D_OPT_PATH)
    out_name = param['out_put'] + '.csv'
    D_OPT.to_csv(os.path.join(D_OPT_PATH, out_name))


if __name__ == '__main__':

    try:
        args = init_param()
        params = vars(args)
        trail_id = params['id']
        start_time = str(time.asctime())
        if not os.path.exists('./log/'):
            os.mkdir('./log/')
        if not os.path.exists('./log/'):
            os.mkdir('./log/')
        if not os.path.exists('./log/' + trail_id):
            os.mkdir('./log/' + trail_id)
        if not os.path.exists('./log/' + trail_id + '/' +
                              params['name']):
            os.mkdir('./log/' + trail_id + '/' + params['name'])
        log_file = './log/' + trail_id + '/' + params['name'] + '/' + start_time + '.log'
        logging.basicConfig(filename=log_file, level=logging_level[params[
            'log_level']], format=
                            '%(asctime)s - %(levelname)s : %(message)s', datefmt=
                            '%Y/%m/%d %H:%M:%S')
        logger = logging.getLogger('')
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp/')
        if not os.path.exists('./tmp/' + params['name'] + '/'):
            os.mkdir('./tmp/' + params['name'] + '/')
        info(params)
        train(params)
    except Exception as exception:
        error(exception)
        raise
