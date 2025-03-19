import os
import warnings

import numpy as np

import DataLoader
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import Neural
import Util

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Presenting tests results
y_train = DataLoader.load('UCI HAR Dataset/train/y_train.txt')
y_test = DataLoader.load('UCI HAR Dataset/test/y_test.txt')
fig, axes = plt.subplots(2, 1, figsize=(20, 10))
axes[0].plot([i for i in range(len(y_train))], y_train)
axes[0].set_ylabel('activity')
axes[0].set_xlabel('time')
axes[0].set_title('training set activity graph')
axes[1].plot([i for i in range(len(y_test))], y_test)
axes[1].set_ylabel('activity')
axes[1].set_xlabel('time')
axes[1].set_title('test set activity graph')
plt.tight_layout()
Util.save_plot('graphs/activity plots')
plt.close()
# loading & preparing Data
x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
# building net (testing net structure, learning rate and epochs in layer and learning tests)
model = Neural.build_net((561,), 6, [256, 64, 16], function='elu', regularizer='no_reg')
model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy', 'mae'])
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=0)
# using model for classification
loss, accuracy, mae = model.evaluate(x_test, y_test)
print(f'acc={accuracy}, loss={loss}, mae={mae}')
predictions = np.array(Util.unify(model.predict(x_test)))
actual = np.array(Util.unify(y_test))
plt.figure(figsize=(20, 6))
plt.plot([i for i in range(len(y_test))], actual, label="Actual", color='blue')
plt.plot([i for i in range(len(y_test))], predictions, label="Predicted", color='green')
plt.title("actual vs predicted")
plt.xlabel("Time")
plt.ylabel("Action")
plt.legend()
Util.save_plot('graphs/actual vs predicted')
