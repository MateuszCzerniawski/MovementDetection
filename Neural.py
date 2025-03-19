import itertools

import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import Sequential
import tensorflow
import multiprocessing as mp
import Util

pos_epochs = [10, 20, 50, 100, 200]
pos_batches = [16, 32, 64, 128, 256, 512]
pos_unit_counts = [256, 128, 64, 32, 16]
pos_dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
pos_reg_vals = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
pos_learning_rates = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
pos_optimizers = [Adam, SGD, RMSprop]


def categorise(data, classes=6):
    return tensorflow.keras.utils.to_categorical(data - 1, num_classes=classes)


def explore_models(models, loss, accuracy, mae, time):
    l_min = 0
    a_max = 0
    m_min = 0
    t_min = 0
    for i in range(1, len(models)):
        l_min = i if loss[i] < loss[l_min] else l_min
        a_max = i if accuracy[i] > accuracy[a_max] else a_max
        m_min = i if mae[i] < mae[m_min] else m_min
        t_min = i if time[i] < time[t_min] else t_min
    return (models[l_min], loss[l_min]), (models[a_max], accuracy[a_max]), (models[m_min], mae[m_min]), (
        models[t_min], time[t_min])


def combine_fits():
    return [(e, b) for b in pos_batches for e in pos_epochs]


def combine_optimisers():
    return [(o, lr) for lr in pos_learning_rates for o in pos_optimizers]


def combine_nets(regularizer_vals=None, dropouts=None):
    def descending_arrays(array):
        def is_descending(arr):
            return all(arr[i] > arr[i + 1] for i in range(len(arr) - 1))

        descending = []
        for i in range(len(array)):
            descending.extend([list(arr) for arr in itertools.combinations_with_replacement(array, i + 1)])
        descending = [arr for arr in descending if is_descending(arr)]
        return descending

    tmp = []
    layers = descending_arrays(pos_unit_counts)
    if regularizer_vals == 'various':
        for units in layers:
            tmp.extend([[units, list(reg_vals)] for reg_vals in itertools.combinations(pos_reg_vals, len(units))])
    elif regularizer_vals == 'uniform':
        for units in layers:
            reg_vals = []
            for val in pos_reg_vals:
                reg_vals.append([val for i in range(len(units))])
            tmp.extend([units, vals] for vals in reg_vals)
    else:
        tmp.extend([[units, []] for units in layers])
    nets = []
    for config in tmp:
        drops = []
        if dropouts == 'various':
            drops = [drop for drop in itertools.combinations(pos_dropouts, len(config[0]))]
        elif dropouts == 'uniform':
            for d in pos_dropouts:
                drops.append([d for i in range(len(config[0]))])
        for drop in drops:
            nets.append({'units': config[0], 'reg_vals': config[1], 'dropouts': drop})
        if len(drops) == 0:
            nets.append({'units': config[0], 'reg_vals': config[1], 'dropouts': []})
    return nets


def build_net(inputs, output, units_counts, function='relu', dropouts=[], regularizer='l1', regularizer_vals=[]):
    regularizer = l2 if regularizer == 'l2' else l1
    if regularizer == 'no_reg':
        regularizer_vals = []
    dropout_index = 0
    regularize_index = 0
    model = Sequential()
    model.add(keras.Input(shape=inputs))
    model.add(Dense(units_counts[0], activation=function, kernel_regularizer=regularizer(regularizer_vals[0]))
              if len(regularizer_vals) > 0 else Dense(units_counts[0], activation=function))
    regularize_index += 1
    if len(dropouts) > 0:
        model.add(Dropout(dropouts[0]))
        dropout_index += 1
    for units in units_counts[1:]:
        if regularize_index < len(regularizer_vals):
            model.add(Dense(units, activation=function,
                            kernel_regularizer=regularizer(regularizer_vals[regularize_index])))
            regularize_index += 1
        else:
            model.add(Dense(units, activation=function))
        if dropout_index < len(dropouts):
            model.add(Dropout(dropouts[dropout_index]))
            dropout_index += 1
    model.add(Dense(output, activation='softmax',
                    kernel_regularizer=regularizer(regularizer_vals[regularize_index]))
              if regularize_index < len(regularizer_vals) else Dense(output, activation='softmax'))
    return model


def fit_evaluate(input):
    start = Util.measure_time()
    compiled_model, x_train, y_train, x_test, y_test, epochs, batch_size = input
    compiled_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    loss, accuracy, mae = compiled_model.evaluate(x_test, y_test)
    return compiled_model, loss, accuracy, mae, Util.measure_time(start)


def multi_build_net(inputs, output, units_counts, dropouts=[], regularizer_vals=[]):
    return [
        build_net(inputs, output, units_counts, function='relu', dropouts=dropouts, regularizer='l1',
                  regularizer_vals=regularizer_vals),
        build_net(inputs, output, units_counts, function='relu', dropouts=dropouts, regularizer='l2',
                  regularizer_vals=regularizer_vals),
        build_net(inputs, output, units_counts, function='elu', dropouts=dropouts, regularizer='l1',
                  regularizer_vals=regularizer_vals),
        build_net(inputs, output, units_counts, function='elu', dropouts=dropouts, regularizer='l2',
                  regularizer_vals=regularizer_vals)
    ]


def multi_compile_net(model):
    compiled = []
    optimizers = ['adam', 'sgd', 'rmsprop']
    loss_arr = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
    for optimizer in optimizers:
        for loss in loss_arr:
            cloned = tensorflow.keras.models.clone_model(model)
            cloned.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mae'])
            compiled.append(cloned)
    return compiled


def multi_fit_evaluate(model, x_train, y_train, x_test, y_test, epochs=20, batch_size=32):
    models = multi_compile_net(model)
    losses = []
    accuracies = []
    maes = []
    times = []
    inputs = [(compiled, x_train, y_train, x_test, y_test, epochs, batch_size) for compiled in models]
    with mp.Pool(processes=len(models)) as pool:
        for compiled_model, loss, accuracy, mae, time in pool.map(fit_evaluate, inputs):
            losses.append(loss)
            accuracies.append(accuracy)
            maes.append(mae)
            times.append(time)
    return models, losses, accuracies, maes, times


def batch_build_nets(inputs, output, configs, multi=False):
    return [multi_build_net(inputs, output,
                            conf['units'], dropouts=conf['dropouts'], regularizer_vals=conf['reg_vals'])
            for conf in configs] if multi else [build_net(inputs, output,
                                                          conf['units'], dropouts=conf['dropouts'],
                                                          regularizer_vals=conf['reg_vals'])
                                                for conf in configs]


def batch_fit_evaluate(model, x_train, y_train, x_test, y_test):
    batch_models = []
    batch_losses = []
    batch_accuracies = []
    batch_maes = []
    batch_times = []
    for epochs, batch_size in combine_fits():
        models, losses, accuracies, maes, times = multi_fit_evaluate(
            model, x_train, y_train, x_test, y_test,
            epochs=epochs, batch_size=batch_size)
        batch_models.extend(models)
        batch_losses.extend(losses)
        batch_accuracies.extend(accuracies)
        batch_maes.extend(maes)
        batch_times.extend(times)
    return batch_models, batch_losses, batch_accuracies, batch_maes, batch_times
