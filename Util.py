import itertools
import os

import numpy as np
import tensorflow
import time
import pandas as pd

from matplotlib import pyplot as plt


def save_plot(path):
    if not os.path.exists(path):
        plt.savefig(path)


def measure_time(start=None):
    end = time.time()
    return float(f'{(end - start):.1f}') if start is not None else float(f'{end:.1f}')


def format_float(number):
    return float(f'{number:.4f}')


def format_percent(number):
    return float(f'{(number * 100):.2f}')


def find_best(data, target='min'):
    best = 0
    if target == 'min':
        for i in range(1, len(data)):
            best = i if data[i][1] < data[best][1] else best
    if target == 'max':
        for i in range(1, len(data)):
            best = i if data[i][1] > data[best][1] else best
    return data[best]


def describe_model(model, output='both'):
    layers = []
    params = []
    for layer in model.layers:
        layers.append(f'{layer.get_config()}')
    params.append(f'optimizer: {model.optimizer.get_config()['name']}')
    params.append(f'loss_function: {model.loss}')
    return {
        'layers': layers,
        'params': params,
        'both': (layers, params)
    }.get(output)


def estimate_work(array, time_per_unit=40):
    size = len(array)
    estimate = time_per_unit * size / 6
    return f'{size} nets to train, expected time: {estimate:.1f}s'


def plot_measurements(data, path=None, names=None):
    plt.close()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'lime', 'black']
    index = 0
    names = [' ' for i in range(len(data))] if names is None else names
    for series in data:
        color = colors[index] if index < len(colors) else 'black'
        acc_counts = pd.value_counts(pd.cut(series['accuracy'], bins=10), sort=False)
        loss_counts = pd.value_counts(pd.cut(series['loss'], bins=10), sort=False)
        mae_counts = pd.value_counts(pd.cut(series['mae'], bins=10), sort=False)
        time_counts = pd.value_counts(pd.cut(series['time'], bins=10), sort=False)
        acc_counts.plot(kind='bar', ax=axes[0], position=index, width=0.2, color=color, label=names[index])
        loss_counts.plot(kind='bar', ax=axes[1], position=index, width=0.2, color=color, label=names[index])
        mae_counts.plot(kind='bar', ax=axes[2], position=index, width=0.2, color=color, label=names[index])
        time_counts.plot(kind='bar', ax=axes[3], position=index, width=0.2, color=color, label=names[index])
        index += 1
    axes[0].set_title('accuracy')
    axes[1].set_title('loss')
    axes[2].set_title('mae')
    axes[3].set_title('time')
    if names is not None:
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[3].legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        save_plot(path)


def unify(tab):
    unified = []
    for arr in tab:
        index = 0
        tmp = arr[index]
        for i in range(1, len(arr)):
            if arr[i] > tmp:
                index = i
                tmp = arr[index]
        unified.append(index)
    return unified
