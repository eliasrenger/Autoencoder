# external imports
import numpy as np
import pandas as pd
from scipy.optimize import root
import tensorflow as tf
from tqdm import tqdm

# internal imports
from autoencoder import Autoencoder
from config import *

np.random.seed(68)
# Function to compute the Jacobian matrix
def compute_jacobian(model, inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        encoded = model.encode(inputs)
        outputs = model.decode(encoded)
    jacobian = tape.jacobian(outputs, inputs)
    return jacobian, outputs

def compute_jacobian_diff(model, inputs, delta = 1):
    deltas = delta * np.eye(N_DIM)
    outputs = model.predict(inputs, verbose=0)
    d_outputs = model.predict(inputs + deltas, verbose=0)
    jacobian = (d_outputs - outputs) / delta
    return jacobian, outputs


def find_fixed_points(model, initial_guess=np.array([np.nan, np.nan, np.nan]), tol=1e-6, n_points=10, max_iter=50, verbose=0):
    all_fixed_points = []
    error = []
    
    def objective(x):
        x=x.reshape(-1, N_DIM)
        return np.squeeze(model.predict(x, verbose=0) - x)
    
    def generate_guess():
        low = -5
        high = 5
        return np.random.uniform(low=low, high=high, size=(N_DIM))

    if np.any(np.isnan(initial_guess)):                            
        current_guess = generate_guess()
    else:
        current_guess = initial_guess

    iter_count = 0
    while len(all_fixed_points) < n_points and iter_count < max_iter:
        iter_count += 1
        sol = root(objective, current_guess, method='lm', tol=tol)
        current_guess = sol.x
        if sol.success:
            err = np.linalg.norm(objective(current_guess))
            if err < tol:
                all_fixed_points.append(current_guess)
                error.append(err)
                if verbose == 1:
                    print("Found fixed point:", current_guess)
        else:
            if verbose == 1:
                print("Root-finding algorithm failed.")
        current_guess = generate_guess()
    if iter_count == max_iter:
        print("Maximum iterations reached.")

    return np.array(all_fixed_points), error

# Example usage:
if __name__ == "__main__":
    # Load the model
    model = Autoencoder(N_DIM, 
                        ENCODED_DIM, 
                        hidden_layers=HIDDEN_LAYERS, 
                        learning_rate=LEARNING_RATE, 
                        optimizer=OPTIMIZER,
                        )
    columns = ['x', 'y', 'epoch', 'batch', 'difference']
    df = pd.DataFrame(columns=columns)
    n_batches = int(np.ceil(N_POINTS / BATCH_SIZE))
    start_epoch = 1
    end_epoch = N_EPOCHS + 1
    start_batch = n_batches
    end_batch = n_batches + 1
    for epoch in tqdm(range(start_epoch, end_epoch)):
        for batch in range(start_batch, end_batch):
            model_id = f'{epoch:03d}-{batch:03d}'
            model_path = f'models/{MODEL_NAME}/cp-{model_id}.weights.h5'
            model._autoencoder.load_weights(model_path)
            fixed_points, error = find_fixed_points(model, 
                                                    n_points=N_FIXED_POINTS,
                                                    max_iter = 25,
                                                    verbose = 0, 
                                                    tol = FIXED_POINT_TOL)
            new_df = pd.DataFrame(fixed_points, columns=columns[0:2])
            new_df['difference'] = error
            new_df['batch'] = batch
            new_df['epoch'] = epoch
            df = pd.concat([df, new_df], ignore_index=True)
    #df.to_csv(f'data/{MODEL_NAME}_fixed.csv', index=False)
    df.to_csv('test.csv')