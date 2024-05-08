# external imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# internal imports
from autoencoder import Autoencoder



def visualize(data, color, title):
    ax = plt.figure().add_subplot(projection='3d')
    for id, (point, error) in enumerate(data):
        ax.quiver(point[:,0], point[:,1], point[:,2], 
            error[:, 0], error[:, 1], error[:, 2], normalize = True,
            color=color[id])
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# load model
data_name = 'balls_10000_points'
add = ''
model_id = '0030'
model_path = f'models/{add}{data_name}/cp-{model_id}.weights.h5'

model = Autoencoder(3, 2, hidden_layers=[3], learning_rate=0.001)
model._autoencoder.load_weights(model_path)
weights = model._autoencoder.get_weights()
print(weights)

# load test data
x_test = np.load(f'data/{data_name}.npz')['test_data']

# 3d grid
grid_width = 8
low = -5
high = 5
grid_intervals = np.array([[low, high],
                           [low, high],
                           [low, high]
])
grid_width = 6
grid = np.array(np.meshgrid(np.linspace(grid_intervals[0,0], grid_intervals[0,1], grid_width),
                            np.linspace(grid_intervals[1,0], grid_intervals[1,1], grid_width),
                            np.linspace(grid_intervals[2,0], grid_intervals[2,1], grid_width),
                            sparse=False))
grid = grid.T.reshape(-1, 3)

pred_test = model.predict(x_test)
test_err = pred_test - x_test
pred_grid = model.predict(grid)
grid_err = pred_grid - grid
tot_err = round(np.linalg.norm(test_err), 2)
data = [(x_test, test_err), (grid, grid_err)]
# calculate error
color = ['r', 'b']
title = f'Total error {tot_err}'
parameters = model._autoencoder.get_weights()
n_layers = len(parameters)//2
weights = [parameters[i] for i in range(0, n_layers, 2)]
biases =   [parameters[i] for i in range(1, n_layers, 2)]
visualize(data, color, title)