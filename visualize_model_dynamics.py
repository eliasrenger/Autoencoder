# external imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# internal imports
from autoencoder import Autoencoder
#from AE_training import Autoencoder

# visualize
def visualize(grid, ae_dynamics, error, stable_fixed_points, unstable_fixed_points):
        # set up
        title = f'Total error {error}'
        color = 'b'

        # plot
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(grid[:,0], grid[:,1], grid[:,2], 
                ae_dynamics[:, 0], ae_dynamics[:, 1], ae_dynamics[:, 2],
                normalize=False, color=color)
        if unstable_fixed_points.shape[0] > 0:
             ax.scatter(*unstable_fixed_points.T, color = 'r')
        if stable_fixed_points.shape[0] > 0:
             ax.scatter(*stable_fixed_points.T, color = 'g')

        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

if __name__ == '__main__':
        # load model
        # data_name = 'sin_exp_pol_3d_100_points'
        data_name = 'helix_10000_points'
        add = 'extra3_'
        model_id = '0003'
        model_path = f'models/{add}{data_name}/cp-{model_id}.weights.h5'

        model = Autoencoder(3, 5, hidden_layers=[3])
        model._autoencoder.load_weights(model_path)
        model._autoencoder.summary()

        # load test data
        x_test = np.load(f'data/{data_name}.npz')['test_data']

        # create point grid
        low = 8
        high = 15
        grid_intervals = np.array([[low, high],
                                   [low, high],
                                   [low, high]
        ])
        grid_width = 7
        grid = np.array(np.meshgrid(np.linspace(grid_intervals[0,0], grid_intervals[0,1], grid_width),
                                    np.linspace(grid_intervals[1,0], grid_intervals[1,1], grid_width),
                                    np.linspace(grid_intervals[2,0], grid_intervals[2,1], grid_width),
                                    sparse=False))
        grid = grid.T.reshape(-1, 3)

        # study autoencoder operating on grid
        reconstructed_grid = model.predict(grid)
        ae_dynamics = reconstructed_grid - grid

        # study autoencoder operating on test data
        x_test_reconstructed = model.predict(x_test)
        error = np.linalg.norm(x_test_reconstructed - x_test)
        error = round(error,2)

        # load fixed points
        d_path = f'models/fixed_points/{add}{data_name}_cp-{model_id}.npz'
        try:
            arr = np.load(d_path)
            stable_fixed_points = arr['stable_fixed_points']
            unstable_fixed_points = arr['unstable_fixed_points']
        except:
            stable_fixed_points, unstable_fixed_points = None, None

        visualize(grid, ae_dynamics, error, stable_fixed_points, unstable_fixed_points)