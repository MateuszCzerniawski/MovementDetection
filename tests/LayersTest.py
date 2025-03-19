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


def test_net(input):
    build, params, data = input
    model = Neural.build_net(build[0], build[1], build[2],
                             function=params[0], regularizer=params[1], regularizer_vals=build[3])
    model.compile(optimizer=params[2], loss=params[3], metrics=['accuracy', 'mae'])
    start = Util.measure_time()
    model.fit(data[0], data[1], epochs=20, batch_size=32, verbose=0)
    loss, accuracy, mae = model.evaluate(data[2], data[3])
    output = [accuracy, loss, mae, Util.measure_time(start)]
    while len(build[2]) < 5:
        build[2].append(0)
    for i in range(len(build[2])):
        output.insert(i, build[2][i])
    return tuple(output)


if __name__ == '__main__':
    test_path = '../tests results/layers test'
    configs = Neural.combine_nets()
    x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
    results = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': [], 'layer5': [],
               'accuracy': [], 'loss': [], 'mae': [], 'time': []}
    results = pd.DataFrame(results)
    if not os.path.exists(test_path):
        inputs = [(((561,), 6, conf['units'], conf['reg_vals']),
                   ('elu', 'no_reg', 'rmsprop', 'binary_crossentropy'),
                   (x_train, y_train, x_test, y_test))
                  for conf in configs]
        tmp = []
        with mp.Pool(processes=6) as pool:
            for record in pool.map(test_net, inputs):
                tmp.append(record)
        for record in tmp:
            results.loc[-1] = record
            results.index = results.index + 1
            results = results.sort_index()
        results.to_csv(test_path, index=False)
    else:
        results = pd.read_csv(test_path)
    Util.plot_measurements([results], path='graphs/layers test/general')
    devided = [pd.DataFrame({'layer1': [], 'layer2': [], 'layer3': [], 'layer4': [], 'layer5': [],
                             'accuracy': [], 'loss': [], 'mae': [], 'time': []}) for i in range(5)]
    for index, row in results.iterrows():
        size = -1
        for i in range(1, 6):
            size += 1 if row[f'layer{i}'] != 0 else 0
        devided[size].loc[-1] = row
        devided[size].index = devided[size].index + 1
        devided[size] = devided[size].sort_index()
    Util.plot_measurements(devided, names=[f'{i} layers' for i in range(1, 6)], path='graphs/layers test/layers counts')
    by_first_layer = [results[results['layer1'] == i] for i in Neural.pos_unit_counts]
    Util.plot_measurements(by_first_layer, names=[f'layer1={i}' for i in Neural.pos_unit_counts], path='graphs/layers test/first layer')
