import warnings
import os

import DataLoader

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import Neural
import Util

if __name__ == '__main__':
    loss = []
    accuracy = []
    mae = []
    time = []
    x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
    model_builds = Neural.multi_build_net((561,), 6, [256, 128, 64, 32, 16],
                                          regularizer_vals=[0.01, 0.01, 0.01, 0.01, 0.01])
    for model in model_builds:
        models, l, a, m, t = Neural.multi_fit_evaluate(model, x_train, y_train, x_test, y_test)
        best_loss, best_accuracy, best_mae, best_time = Neural.explore_models(models, l, a, m, t)
        loss.append(best_loss)
        accuracy.append(best_accuracy)
        mae.append(best_mae)
        time.append(best_time)

    best_loss = Util.find_best(loss)
    best_accuracy = Util.find_best(accuracy, target='max')
    best_mae = Util.find_best(mae)
    best_time = Util.find_best(time)
    print(f'Best loss: {Util.format_percent(best_loss[1])}')
    for line in Util.describe_model(best_loss[0], output='params'):
        print(line)
    print(f'Best accuracy: {Util.format_percent(best_accuracy[1])}%')
    for line in Util.describe_model(best_accuracy[0], output='params'):
        print(line)
    print(f'Best mae: {Util.format_float(best_mae[1])}')
    for line in Util.describe_model(best_mae[0], output='params'):
        print(line)
    print(f'Best time: {best_time[1]}s')
    for line in Util.describe_model(best_time[0], output='params'):
        print(line)
