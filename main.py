from DQNPlayer import DQNPlayer
from QPlayer import QPlayer, RandomPlayer
from Game import Game
import datetime
import time
import numpy as np
import itertools
import pandas as pd
from multiprocessing import dummy as multiprocessing

def expand_space(param_dict, keys=None, params=None):
    if keys is None:
        keys = list(param_dict.keys())
    if params is None:
        params = {}
    if len(keys) > 0:
        key = keys[0]
        for param in param_dict[key]:
            new_params = params.copy()
            new_params[key] = param
            yield from expand_space(param_dict, keys[1:], new_params)
    else:
        yield params

op_dict = {
    'random': RandomPlayer(),
    'qt1000': QPlayer(),
    'qt6000': QPlayer(),
    'dqn1000': DQNPlayer(),
    'dqn6000': DQNPlayer(),
}
op_dict_loose = {
    'random': RandomPlayer(),
    'qt1000': QPlayer(),
    'qt6000': QPlayer(),
    'dqn1000': DQNPlayer(),
    'dqn6000': DQNPlayer(),
}


def runner(params):
    loose_on_invalid = params['loose_on_invalid']
    my_op_dict = op_dict if not loose_on_invalid else op_dict_loose
    lr = params['lr']
    loss = params['loss'] if 'loss' in params else None
    player1 = QPlayer(lr) if params['player1'] == 'qt' else DQNPlayer(lr, loss)
    player2 = my_op_dict[params['player2']]
    pretrain_op = my_op_dict[params['pretrain_p1']]
    pretrain_p1_length = params['pretrain_p1_length']

    pretrain_log = None
    if pretrain_p1_length > 0:
        pretrain_log, grouped_scores = Game.play_multigame(pretrain_p1_length, [player1, pretrain_op], [1, 2], [True, False], alternate=True)

    log1, gs1 = Game.play_multigame(1000, [player1, player2], [1, 2], [True, False], alternate=True, loose_on_invalid=loose_on_invalid)
    log2, gs2 = Game.play_multigame(1000, [player1, player2], [1, 2], [False, True], alternate=False, loose_on_invalid=loose_on_invalid)
    log3, gs3 = Game.play_multigame(1000, [player2, player1], [2, 1], [False, True], alternate=False, loose_on_invalid=loose_on_invalid)

    [i.update({'test': 'm1', 'loose_on_invalid': loose_on_invalid}) for i in log1]
    [i.update({'test': 'm2', 'loose_on_invalid': loose_on_invalid}) for i in log2]
    [i.update({'test': 'm3', 'loose_on_invalid': loose_on_invalid}) for i in log3]

    [i.update(params) for i in log1]
    [i.update(params) for i in log2]
    [i.update(params) for i in log3]

    return itertools.chain(log1, log2, log3)


if __name__ == '__main__':
    print('1')
    logA, gsA = Game.play_multigame(1000, [op_dict['random'], op_dict['qt1000']], [1, 2], [False, True], alternate=True, loose_on_invalid=False)
    print('2')
    logB, gsB = Game.play_multigame(6000, [op_dict['random'], op_dict['qt6000']], [1, 2], [False, True], alternate=True, loose_on_invalid=False)
    # print('3')
    # logC, gsC = Game.play_multigame(1000, [op_dict['random'], op_dict['dqn1000']], [1, 2], [False, True], alternate=True, loose_on_invalid=False)
    # print('4')
    # logD, gsD = Game.play_multigame(6000, [op_dict['random'], op_dict['dqn6000']], [1, 2], [False, True], alternate=True, loose_on_invalid=False)
    print('5')
    logE, gsE = Game.play_multigame(1000, [op_dict_loose['random'], op_dict_loose['qt1000']], [1, 2], [False, True], alternate=True, loose_on_invalid=True)
    print('6')
    logF, gsF = Game.play_multigame(6000, [op_dict_loose['random'], op_dict_loose['qt6000']], [1, 2], [False, True], alternate=True, loose_on_invalid=True)
    # print('7')
    # logG, gsG = Game.play_multigame(1000, [op_dict_loose['random'], op_dict_loose['dqn1000']], [1, 2], [False, True], alternate=True, loose_on_invalid=True)
    # print('8')
    # logH, gsH = Game.play_multigame(6000, [op_dict_loose['random'], op_dict_loose['dqn6000']], [1, 2], [False, True], alternate=True, loose_on_invalid=True)
    # print('9')

    # DQNPlayer will output tensorboard data to ./tfb_data/*
    # 1. 'pip install tensorboard'
    # 2. in terminal go to TicTacToeRLProject folder
    # 3. 'tensorboard --logdir tfb_data' will start http server with plots
    # 4. open 127.0.0.1:6006 (not 0.0.0.0:6006) in internet browser

    timestamp = datetime.datetime.now()
    params_space1 = {
        'player1': [
            'dqn',
        ],
        'player2': [
            'random',
            'qt1000',
            'qt6000',
            # 'dqn1000',
            # 'dqn6000',
        ],
        'pretrain_p1': [
            'random',
            'qt1000',
            'qt6000',
            # 'dqn1000',
            # 'dqn6000',
        ],
        'pretrain_p1_length': [
            0,
            1000,
            2000,
            5000,
        ],
        'lr': [
            # 0.1,
            # 0.01,
            # 0.001,
            0.0001,
        ],
        'loss': [
            'huber',
            # 'mse',
        ],
        'loose_on_invalid': [
            True,
            False
        ],
    }
    params_space2 = {
        'player1': [
            'qt',
        ],
        'player2': [
            'qt1000',
            'qt6000',
            # 'dqn1000',
            # 'dqn6000',
            'random',
        ],
        'pretrain_p1': [
            'random',
            'qt1000',
            'qt6000',
            # 'dqn1000',
            # 'dqn6000',
        ],
        'pretrain_p1_length': [
            0,
            1000,
            2000,
            5000,
        ],
        'lr': [
            0.1,
            # 0.01,
            # 0.001,
            # 0.0001,
        ],
        'loose_on_invalid': [
            True,
            False
        ],
    }
    params_space_size = np.prod([len(v) for v in params_space1.values()]) + np.prod([len(v) for v in params_space2.values()])
    name = str(timestamp).replace(' ', '_').replace('-', '.').replace(':', '.')

    num_threads = 1
    time_start = time.time()
    pool = multiprocessing.Pool(num_threads)
    log = pd.DataFrame()
    cache = []
    cache_size = 1  # 3000*20
    print('start')
    for i, v in enumerate(pool.imap(runner, itertools.chain(expand_space(params_space2), expand_space(params_space1)), chunksize=1)):
        if v is not None:
            cache.extend(v)
            print('{:4d}/{:4d}, {:3.3f}s'.format(i, params_space_size, time.time() - time_start))
            if len(cache) >= cache_size:
                log = log.append(cache, ignore_index=True)
                cache.clear()
                log.to_csv(name + '_{:04d}of{:04d}.csv'.format(i, params_space_size))
                print('flushed')
    if len(cache) > 0:
        log = log.append(cache, ignore_index=True)
        cache.clear()
        log.to_csv(name + '.csv')
        print('flushed2')

    # full flush
    log.to_csv(name + '.csv')

###########################
    # random = RandomPlayer()
    # dqn = DQNPlayer()
    # pl = QPlayer()
    #
    # Game.play_multigame(10000, [random, pl], [1, 2], [False, True], alternate=True)
    # Game.play_multigame(1000, [random, pl], [1, 2])
    # Game.play_multigame(1000, [pl, random], [2, 1])
    # Game.play_multigame(1000, [random, pl], [1, 2], alternate=True)
    # print('#########################')
    # Game.play_multigame(10000, [dqn, random], [1, 2], [True, False], alternate=True)
    # Game.play_multigame(10000, [dqn, pl], [1, 2], [True, False], alternate=True)
    # Game.play_multigame(1000, [dqn, pl], [1, 2])
    # Game.play_multigame(1000, [pl, dqn], [2, 1])
    # Game.play_multigame(1000, [dqn, pl], [1, 2], alternate=True)
    # print('#########################')
    # Game.play_multigame(1000, [dqn, random], [1, 2])
    # Game.play_multigame(1000, [random, dqn], [2, 1])
    # Game.play_multigame(1000, [dqn, random], [1, 2], alternate=True)
