"""
Define the data space and model architecture so that all scripts can be run with the same configuration.
"""

# data
N_DIM = 2
STRUCTURE = 'sin'
N_POINTS = 200
TRAIN_TEST_SPLIT = 0.8
N_TOT_POINTS = int(N_POINTS / TRAIN_TEST_SPLIT)
DATA_NAME = f'{STRUCTURE}_{N_DIM}d_{N_POINTS}_points'
DATA_PATH = f'data/{DATA_NAME}.npz'

# model
ENCODED_DIM = 1
HIDDEN_LAYERS = [2]
OPTIMIZER = 'sgd'
LEARNING_RATE = 0.01
BATCH_SIZE = 32
N_EPOCHS = 300
MODEL_NAME = f'{DATA_NAME}_{OPTIMIZER}_{LEARNING_RATE}_last'
MODEL_PATH = f'models/{MODEL_NAME}'

# fixed point
FIXED_POINT_TOL = 1e-6
N_FIXED_POINTS = 15