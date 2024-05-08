# external imports
import numpy as np
from scipy.optimize import root
from numpy.linalg import eig
import tensorflow as tf
import matplotlib.pyplot as plt

# internal imports
from autoencoder import Autoencoder
N_DIM = 2

np.random.seed(42)
tf.random.set_seed(42)

def compute_jacobian_diff(model, inputs, delta = 0.0001):
    deltas = delta * np.eye(N_DIM)
    inputs = np.tile(inputs, N_DIM).reshape(-1, N_DIM, N_DIM)
    lower_inputs, upper_inputs = inputs - deltas, inputs + deltas
    lower_inputs = lower_inputs.reshape(-1, N_DIM)
    upper_inputs = upper_inputs.reshape(-1, N_DIM)
    lower_outputs = model.predict(lower_inputs, verbose = 0).reshape(-1, N_DIM, N_DIM)
    upper_outputs = model.predict(upper_inputs, verbose = 0).reshape(-1, N_DIM, N_DIM)
    lower_outputs = np.transpose(lower_outputs, axes=(0, 2, 1))
    upper_outputs = np.transpose(upper_outputs, axes=(0, 2, 1))
    jacobian = (upper_outputs - lower_outputs) / (2*delta)
    return jacobian

def find_orbit_diff(model, period_length, tol=1e-5, n_rand_points = 10, max_iter=10, step_scale = 0.1, verbose = 1):
    periodic_orbits = []
    message = []
    return_dict = {'found': 'placeholder',
                'periodic_orbits': periodic_orbits,
                'message': message
                }
    
    def generate_guess(period_length):
        # create initial guess
        spread = 4
        center = np.array([0, 1])
        current_guess = center + np.random.uniform(low=-spread, high=spread, size=(period_length, N_DIM))
        return current_guess
    for guess_id in range(n_rand_points):
        x_current = generate_guess(period_length)
        for iter in range(max_iter):
            x_new = model.predict(x_current, verbose=0)
            F = np.roll(x_current, -1, axis=0) - x_new

            # compute update
            derived_f = compute_jacobian_diff(model, x_current)
            F_prim = np.zeros((period_length, period_length, N_DIM, N_DIM))
            identity = np.eye(N_DIM)
            for id in range(period_length-1):
                F_prim[id, id] = -derived_f[id]
                F_prim[id, id+1] = identity
            F_prim[-1, -1] = -derived_f[-1]
            F_prim[-1, 0] = identity

            # reshape to correct structure in two dimensions
            F_prim = np.transpose(F_prim, axes=(0, 2, 1, 3))
            F_prim = F_prim.reshape(N_DIM*period_length, N_DIM*period_length)
            F = F.flatten()
            update = np.linalg.solve(F_prim, F).reshape(-1, N_DIM)   
            N_x = x_current - step_scale * update
        
            if np.sum(abs(N_x-x_current)) < tol:
                x_current = N_x
                if period_length > 1 and np.sum(abs(x_current[0]-x_current[1])) < tol: # check if fixed point found
                    message.append('Fixed point found')
                    if verbose == 1:
                        print(f'Orbit length {period_length}: Fixed point found after {iter} iterations, point {guess_id}')
                    break
                else:
                    periodic_orbits.append(x_current)
                    message.append('Successful')
                    x_current = generate_guess(period_length)
                    if verbose == 1:
                        print(f'Orbit length {period_length}: Successfully found periodic orbit of length {period_length} after {iter} iterations, point {guess_id}')
                    break
            elif np.max(x_current) > 1e10:
                if verbose == 1:
                    print(f'Orbit length {period_length}: Newton method diverged for {period_length}.')
                message.append('Diverging')
                break
            else:
                x_current = N_x #+ np.random.normal(0, 1e-2, x_current.shape)

    if len(periodic_orbits) == 0:
        if verbose == 1:
            print(f"Maximum iterations reached without finding periodic orbit of length {period_length}.")
        message.append('Failed to find any periodic orbits')
        return_dict['found'] = False
    else:
        return_dict['found'] = True
    return return_dict

if __name__ == "__main__":
    # define model
    data_name = 'sin_2d_10000_points'
    model_id = '0030'
    add = ''
    model_path = f'models/{add}{data_name}/cp-{model_id}.weights.h5'

    # Load the model
    model = Autoencoder(N_DIM, 1, hidden_layers=[2])
    model._autoencoder.load_weights(model_path)

    # Find fixed points
    orbit_results = []
    base_tol = 1e-1
    for orbit_length in range(2, 20):
        result = find_orbit_diff(model, orbit_length, tol = orbit_length*base_tol, n_rand_points=20, max_iter=30, step_scale=0.2, verbose = 1)
        if result['found']:
            orbit_results.append(result['periodic_orbits'])
            print('------ Orbit ------')
            print(orbit_results)
