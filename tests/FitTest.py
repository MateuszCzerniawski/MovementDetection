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


def test_fit(input):
    build, params, data = input
    model = Neural.build_net(build[0], build[1], build[2],
                             function='elu', regularizer='no_reg', regularizer_vals=build[3])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'mae'])
    start = Util.measure_time()
    model.fit(data[0], data[1], epochs=params[0], batch_size=params[1], verbose=0)
    loss, accuracy, mae = model.evaluate(data[2], data[3])
    return params[0], params[1], accuracy, loss, mae, Util.measure_time(start)


if __name__ == '__main__':
    test_path = '../tests results/fit test'
    x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
    results = {'epochs': [], 'batch_size': [], 'accuracy': [], 'loss': [], 'mae': [], 'time': []}
    results = pd.DataFrame(results)
    if not os.path.exists(test_path):
        inputs = [(((561,), 6, [256, 64, 16], []), params, (x_train, y_train, x_test, y_test)) for params in
                  Neural.combine_fits()]
        tmp = []
        with mp.Pool(processes=6) as pool:
            for record in pool.map(test_fit, inputs):
                tmp.append(record)
        for record in tmp:
            results.loc[-1] = record
            results.index = results.index + 1
            results = results.sort_index()
        results.to_csv(test_path, index=False)
    else:
        results = pd.read_csv(test_path)
    Util.plot_measurements([results], path='graphs/fit test/general')
    by_epochs = [results[results['epochs'] == e] for e in Neural.pos_epochs]
    Util.plot_measurements(by_epochs, names=[f'{e} epochs' for e in Neural.pos_epochs],
                           path='graphs/fit test/by epochs')
    by_batch = [results[results['batch_size'] == b] for b in Neural.pos_batches]
    Util.plot_measurements(by_batch, names=[f'{b} batches' for b in Neural.pos_batches],
                           path='graphs/fit test/by batches')
