# external imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

# internal imports
from autoencoder import Autoencoder
from config import *

# visualize
def visualize(point, vec, error):
        # set up
        title = f'Total error {error}'
        color = ('r', 'b')
        for id, v in enumerate(vec):
                mean_mag = np.mean(np.linalg.norm(v, axis=1))
                vec[id] = v / mean_mag

        # plot
        ax = plt.figure().add_subplot()
        for id, v in enumerate(vec):
                ax.quiver(point[id][:,0], point[id][:,1], 
                        v[:, 0], v[:, 1], color=color[id])

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.show()

if __name__ == '__main__':
        # load model
        # data_name = 'sin_exp_pol_3d_100_points'
        data_name = DATA_NAME
        model_id = '002-005'
        model_path = f'models/{data_name}/cp-{model_id}.weights.h5'
        print(model_path)
        model = Autoencoder(2, ENCODED_DIM, hidden_layers=HIDDEN_LAYERS, learning_rate=0.001)
        model._autoencoder.load_weights(model_path)
        model._autoencoder.summary()

        # load test data
        x_test = np.load(f'data/{data_name}.npz')['test_data']
        fixed_points = pd.read_csv(f'data/analyzed_{data_name[:-7]}_points.csv')

        # study autoencoder operating on test data
        x_test_reconstructed = model.predict(x_test)
        x_vec = x_test_reconstructed - x_test
        error = np.linalg.norm(x_vec)
        error = round(error, 2)
        print(x_test)

        # create point grid
        low = -5
        high = 5
        grid_intervals = np.array([[low, high],
                                   [low, high]
        ])
        grid_width = 20
        grid = np.array(np.meshgrid(np.linspace(grid_intervals[0,0], grid_intervals[0,1], grid_width),
                                    np.linspace(grid_intervals[1,0], grid_intervals[1,1], grid_width),
                                    sparse=False))
        grid = grid.T.reshape(-1, 2)

        # study autoencoder operating on grid
        reconstructed_grid = model.predict(grid)
        ae_dynamics = reconstructed_grid - grid

        # load test
        point = [x_test, grid]
        vec = [x_vec, ae_dynamics]
        visualize(point, vec, error)