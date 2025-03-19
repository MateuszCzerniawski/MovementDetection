import warnings
import os

import DataLoader

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import Neural
import Util

x_train, y_train, x_test, y_test, labels = DataLoader.load_all()
# single
model = Neural.build_net((561,), 6, [256, 128, 64, 32, 16], dropouts=[0.5, 0.2, 0.1],
                         regularizer_vals=[0.02, 0.01, 0.01])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mae'])
print(f'Test for single net\n{Util.estimate_work([model])}')
# profound test for single net
model_builds = Neural.multi_build_net((561,), 6, [256, 128, 64, 32, 16], dropouts=[0.5, 0.2, 0.1],
                                      regularizer_vals=[0.02, 0.01, 0.01])
compilations = []
for setup in Neural.combine_fits():
    for build in model_builds:
        compilations.extend(Neural.multi_compile_net(build))
print(f'Profound test for single net\n{Util.estimate_work(compilations)}')
# nets
print('nets')
pure = Neural.combine_nets()
print(f'pure nets to test:{Util.estimate_work(pure)}')
reg_uni = Neural.combine_nets(regularizer_vals='uniform')
print(f'with uniform regularization:{Util.estimate_work(reg_uni)}')
reg_var = Neural.combine_nets(regularizer_vals='various')
print(f'with various regularization:{Util.estimate_work(reg_var)}')
d_uni = Neural.combine_nets(dropouts='uniform')
print(f'with uniform dropouts:{Util.estimate_work(d_uni)}')
d_var = Neural.combine_nets(dropouts='various')
print(f'with various dropouts:{Util.estimate_work(d_var)}')
dr_uni = Neural.combine_nets(dropouts='uniform', regularizer_vals='uniform')
print(f'with uniform dropouts and regularization:{Util.estimate_work(dr_uni)}')
dr_var = Neural.combine_nets(dropouts='various', regularizer_vals='various')
print(f'with various dropouts and regularization:{Util.estimate_work(dr_var)}')
total = 6671 * 1080
estimate = 40 * total / 6
in_years = estimate / (60 * 60 * 24*365)
print(f'\nmost profound variant is {total} nets, estimated time{estimate:.1f}s ({in_years:.1f}y)')
