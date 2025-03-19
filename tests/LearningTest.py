import itertools
import warnings
import os
import pandas as pd
import re
from matplotlib import pyplot as plt

import DataLoader
from itertools import product
import multiprocessing as mp

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import Neural
import Util


def test_learning(input):
    build, params, data = input
    model = Neural.build_net(build[0], build[1], build[2],
                             function='elu', regularizer='no_reg', regularizer_vals=build[3])
    model.compile(optimizer=params[0](learning_rate=params[1]), loss='binary_crossentropy', metrics=['accuracy', 'mae'])
    start = Util.measure_time()
    history = model.fit(data[0], data[1], epochs=params[2], batch_size=128, verbose=0,
                        validation_data=(data[2], data[3]))
    pd.DataFrame(history.history).to_csv(
        f'../tests results/histories/{model.optimizer.get_config()['name']}({params[1]}) e={params[2]}')
    loss, accuracy, mae = model.evaluate(data[2], data[3])
    return params[0], params[1], params[2], accuracy, loss, mae, Util.measure_time(start)


if __name__ == '__main__':
    test_path = '../tests results/learning test'
    x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
    results = {'optimiser': [], 'learning': [], 'epochs': [], 'accuracy': [], 'loss': [], 'mae': [], 'time': []}
    results = pd.DataFrame(results)
    if not os.path.exists(test_path):
        inputs = [(((561,), 6, [256, 64, 16], []),
                   (o[0], o[1], e),
                   (x_train, y_train, x_test, y_test))
                  for o, e in itertools.product(Neural.combine_optimisers(), Neural.pos_epochs)]
        tmp = []
        with mp.Pool(processes=6) as pool:
            for record in pool.map(test_learning, inputs):
                tmp.append(record)
        for record in tmp:
            results.loc[-1] = record
            results.index = results.index + 1
            results = results.sort_index()
        results.to_csv(test_path, index=False)
    else:
        results = pd.read_csv(test_path)
    Util.plot_measurements([results], path='../graphs/learning test/general')
    Util.plot_measurements([results[results['optimiser'] == str(o)] for o in Neural.pos_optimizers]
                           , names=[str(i) for i in Neural.pos_optimizers], path='../graphs/learning test/by optimiser')
    Util.plot_measurements([results[results['learning'] == lr] for lr in Neural.pos_learning_rates]
                           , names=[str(i) for i in Neural.pos_learning_rates],
                           path='../graphs/learning test/by learning rate')
    Util.plot_measurements([results[results['epochs'] == e] for e in Neural.pos_epochs]
                           , names=[str(i) for i in Neural.pos_epochs], path='../graphs/learning test/by epochs')
    dir_path = '../tests results/histories'
    for opt in Neural.pos_optimizers:
        opt_name = re.search(r"\.([^\.']*)'", str(opt).lower()).group()[1:-1]
        print(opt_name)
        histories_dict = dict()
        for i in Neural.pos_epochs:
            histories_dict[str(i)] = []
        for f in os.listdir(dir_path):
            o = re.search(r"^[a-zA-Z]+(?=\()", f).group()
            lr = re.search(r"(?<=\().*?(?=\))", f).group()
            e = re.search(r"(?<=e=)\d+", f).group()
            if opt_name.lower() == o.lower():
                history = pd.read_csv(dir_path + '/' + f)
                histories_dict[str(e)].append((lr, history))
        plt.close()
        fig, axes = plt.subplots(5, 3, figsize=(25, 25))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'lime', 'black']
        index = 0
        for e in histories_dict:
            color = 0
            for lr, hist in histories_dict[e]:
                axes[index, 0].plot(range(int(e)), hist['val_accuracy'], label=f'lr={lr}', color=colors[color])
                axes[index, 1].plot(range(int(e)), hist['val_loss'], label=f'lr={lr}', color=colors[color])
                axes[index, 2].plot(range(int(e)), hist['val_mae'], label=f'lr={lr}', color=colors[color])
                axes[index, 0].set_title('accuracy')
                axes[index, 1].set_title('loss')
                axes[index, 2].set_title('mae')
                axes[index, 0].legend()
                axes[index, 1].legend()
                axes[index, 2].legend()
                color += 1
            index += 1
        plt.tight_layout()
        Util.save_plot(f'../graphs/learning test/opt={opt_name}')
