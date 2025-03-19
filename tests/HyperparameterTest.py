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


def test_params(input):
    build, params, data = input
    model = Neural.build_net(build[0], build[1], build[2],
                             function=params[0], regularizer=params[1], regularizer_vals=build[3])
    model.compile(optimizer=params[2], loss=params[3], metrics=['accuracy', 'mae'])
    start = Util.measure_time()
    model.fit(data[0], data[1], epochs=20, batch_size=32, verbose=0)
    loss, accuracy, mae = model.evaluate(data[2], data[3])
    return params[0], params[1], params[2], params[3], accuracy, loss, mae, Util.measure_time(start)


if __name__ == '__main__':
    test_path = '../tests results/hyperparameters test'
    pos_activation_funcs = ['relu', 'elu']
    pos_regularizers = ['no_reg', 'l1', 'l2']
    pos_optimizers = ['adam', 'sgd', 'rmsprop']
    pos_loss_funcs = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
    params = list(product(pos_activation_funcs, pos_regularizers, pos_optimizers, pos_loss_funcs))
    results = {
        'activation_func': [],
        'regularizer': [],
        'optimizer': [],
        'loss_func': [],
        'accuracy': [],
        'loss': [],
        'mae': [],
        'time': []
    }
    if not os.path.exists(test_path):
        results = pd.DataFrame(results)
        x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
        inputs = [
            (((561,), 6, [256, 64, 16], [0.01, 0.01, 0.01]),
             p,
             (x_train, y_train, x_test, y_test))
            for p in params
        ]
        tmp = []
        with mp.Pool(processes=6) as pool:
            for record in pool.map(test_params, inputs):
                tmp.append(record)
        for record in tmp:
            results.loc[-1] = record
            results.index = results.index + 1
            results = results.sort_index()
        results.to_csv(test_path, index=False)
    else:
        results = pd.read_csv(test_path)
    Util.plot_measurements([results], path='graphs/hyperparameters test/general')
    pairs = [(pos_activation_funcs, 'activation_func'),
             (pos_regularizers, 'regularizer'),
             (pos_optimizers, 'optimizer'),
             (pos_loss_funcs, 'loss_func'), ]
    for arr, name in pairs:
        data = []
        names = []
        for p in arr:
            data.append(results[results[name] == p])
            names.append(p)
        Util.plot_measurements(data, names=names, path=f'graphs/hyperparameters test/{name}')
