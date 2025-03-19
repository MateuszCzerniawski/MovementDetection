import warnings
import os
import pandas as pd
from matplotlib import pyplot as plt

import DataLoader
from itertools import product
import multiprocessing as mp

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import Neural
import Util


def test_regularization(input):
    build, params, data = input
    model = Neural.build_net(build[0], build[1], build[2],
                             function=params[0], regularizer=params[1], regularizer_vals=build[3])
    model.compile(optimizer=params[2], loss=params[3], metrics=['accuracy', 'mae'])
    start = Util.measure_time()
    model.fit(data[0], data[1], epochs=20, batch_size=32, verbose=0)
    loss, accuracy, mae = model.evaluate(data[2], data[3])
    return params[1], build[3][0] if params[1] != 'no_reg' else 0, accuracy, loss, mae, Util.measure_time(start)


if __name__ == '__main__':
    test_path = '../tests results/regularization test'
    configs = Neural.combine_nets(regularizer_vals='uniform')
    x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
    results = {'reg': [], 'val': [], 'accuracy': [], 'loss': [], 'mae': [], 'time': []}
    results = pd.DataFrame(results)
    if not os.path.exists(test_path):
        tmp = []
        for conf in configs:
            if conf['units'][0] == 256 and len(conf['units']) in [2, 3]:
                tmp.append(conf)
        configs = tmp
        inputs = []
        for conf in [(((561,), 6, conf['units'], conf['reg_vals']),
                      ('elu', 'no_reg', 'rmsprop', 'binary_crossentropy'),
                      (x_train, y_train, x_test, y_test))
                     for conf in configs]:
            inputs.append(conf)
        for conf in [(((561,), 6, conf['units'], conf['reg_vals']),
                      ('elu', 'l1', 'rmsprop', 'binary_crossentropy'),
                      (x_train, y_train, x_test, y_test))
                     for conf in configs]:
            inputs.append(conf)
        for conf in [(((561,), 6, conf['units'], conf['reg_vals']),
                      ('elu', 'l2', 'rmsprop', 'binary_crossentropy'),
                      (x_train, y_train, x_test, y_test))
                     for conf in configs]:
            inputs.append(conf)
        tmp = []
        with mp.Pool(processes=6) as pool:
            for record in pool.map(test_regularization, inputs):
                tmp.append(record)
        for record in tmp:
            results.loc[-1] = record
            results.index = results.index + 1
            results = results.sort_index()
        results.to_csv(test_path, index=False)
    else:
        results = pd.read_csv(test_path)
    Util.plot_measurements([results], path='graphs/regularization test/general')
    by_reg = [results[results['reg'] == reg] for reg in ['no_reg', 'l1', 'l2']]
    Util.plot_measurements(by_reg,
                           names=[reg for reg in ['no_reg', 'l1', 'l2']],
                           path='graphs/regularization test/by reg')
    Util.plot_measurements([by_reg[0]], path='graphs/regularization test/no reg')
    Util.plot_measurements([by_reg[1][by_reg[1]['val'] == val] for val in Neural.pos_reg_vals],
                           names=[val for val in Neural.pos_reg_vals],
                           path='graphs/regularization test/l1')
    Util.plot_measurements([by_reg[2][by_reg[2]['val'] == val] for val in Neural.pos_reg_vals],
                           names=[val for val in Neural.pos_reg_vals],
                           path='graphs/regularization test/l2')
