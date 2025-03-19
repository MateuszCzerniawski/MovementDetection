import pandas as pd
fit_test = pd.read_csv('tests results/fit test')
fit_test['layer1'] = 256
fit_test['layer2'] = 64
fit_test['layer3'] = 16
fit_test['layer4'] = 0
fit_test['layer5'] = 0
fit_test['activation_func'] = 'elu'
fit_test['regularizer'] = 'no_reg'
fit_test['val'] = 0
fit_test['optimiser'] = 'rmsprop'
fit_test['loss_func'] = 'binary_crossentropy'
fit_test['learning'] = -1

hyperparameters_test = pd.read_csv('tests results/hyperparameters test')
hyperparameters_test['layer1'] = 256
hyperparameters_test['layer2'] = 64
hyperparameters_test['layer3'] = 16
hyperparameters_test['layer4'] = 0
hyperparameters_test['layer5'] = 0
hyperparameters_test['val'] = 0.01
hyperparameters_test['learning'] = -1
hyperparameters_test['epochs'] = 20
hyperparameters_test['batch_size'] = 32

layers_test = pd.read_csv('tests results/layers test')
layers_test['activation_func'] = 'elu'
layers_test['regularizer'] = 'no_reg'
layers_test['optimiser'] = 'rmsprop'
layers_test['loss_func'] = 'binary_crossentropy'
layers_test['val'] = 0
layers_test['learning'] = -1
layers_test['epochs'] = 20
layers_test['batch_size'] = 32

learning_test = pd.read_csv('tests results/learning test')
learning_test['layer1'] = 256
learning_test['layer2'] = 64
learning_test['layer3'] = 16
learning_test['layer4'] = 0
learning_test['layer5'] = 0
learning_test['activation_func'] = 'elu'
learning_test['regularizer'] = 'no_reg'
learning_test['val'] = 0
learning_test['loss_func'] = 'binary_crossentropy'
learning_test['batch_size'] = 128

regularization_test = pd.read_csv('tests results/regularization test')
regularization_test['layer1'] = 0
regularization_test['layer2'] = 0
regularization_test['layer3'] = 0
regularization_test['layer4'] = 0
regularization_test['layer5'] = 0
regularization_test['activation_func'] = 'elu'
regularization_test['optimiser'] = 'rmsprop'
regularization_test['loss_func'] = 'binary_crossentropy'
regularization_test['learning'] = -1
regularization_test['epochs'] = 20
regularization_test['batch_size'] = 32
regularization_test = regularization_test.rename(columns={'reg': 'regularizer'})

result = pd.concat([fit_test, hyperparameters_test, layers_test, learning_test, regularization_test],
                   axis=0, join='outer', ignore_index=True)
print(result.shape)
result.to_csv('tests results/all')

