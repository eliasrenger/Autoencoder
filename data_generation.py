# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from config import *

np.random.seed(3872)

def func1(X):
    # define the function
    Y = np.zeros((len(X), 3))
    Y[:, 0] = 53*np.sin(X)
    Y[:, 1] = np.exp(0.4*X)
    Y[:, 2] = 3*X**2
    return Y

def func2(X):
    # define the function
    Y = np.zeros((len(X), 3))
    Y[:, 0] = np.sin(X)
    Y[:, 1] = X**2 - 3*X + 2
    Y[:, 2] = X**6 - 3*X**3 + 4
    return Y

def func3(X):
    offset = 0
    Y = np.zeros((len(X), 3))
    Y[:, 0] = offset + np.sin(X)
    Y[:, 1] = offset + np.cos(X)
    Y[:, 2] = offset+X
    return Y

def func4(X):
    # plane
    offset = 0
    n_points = len(X)
    Y = np.zeros((n_points, 3))
    Y[:, 0] = np.random.uniform(0, 10, n_points)
    Y[:, 1] = np.random.uniform(0, 10, n_points)
    Y[:, 2] = offset
    return Y

def func5(X):
    # sin 1d
    offset = 0
    Y = offset + np.sin(X)
    return Y

def func6(X):
    offset = 2
    Y = np.zeros((len(X), 2))
    Y[:, 0] = offset + np.cos(X)
    Y[:, 1] = offset + np.sin(X)
    return Y

def func7(X):
    # sin 2d
    offset = 2
    Y = np.zeros((len(X), 2))
    Y[:,0] = X
    Y[:, 1] = offset + 1.4 * np.sin(X)
    return Y

def func8(X):
    offset = 0
    expansion_rate = 0.5
    Y = np.zeros((len(X), 3))
    Y[:, 0] = offset + expansion_rate * X * np.sin(5*X)
    Y[:, 1] = offset + expansion_rate * X * np.cos(5*X)
    Y[:, 2] = offset + 0.1*X
    return Y

def func9(X):
    def generate_ball(n, r):
        ball = np.empty((n, 3))
        for i in range(n):
            too_big = True
            while too_big:
                X = np.random.uniform(-r, r, size=(3))
                L = np.linalg.norm(X)
                if L < r:
                    too_big = False
            ball[i] = X
        return ball
    
    n_balls = 6
    radii = np.random.uniform(0.2, 1.3, n_balls)
    center = np.random.uniform(-5, 5, (n_balls, 3))
    n_points = int(len(X) / n_balls)
    for id in range(n_balls):
        ball = generate_ball(n_points, radii[id])
        ball += center[id]
        if id == 0:
            Y = ball
        else:
            Y = np.concatenate((Y, ball))
    print('radii', radii)
    print('center', center)
    return Y

def func10(X):
    Y = np.random.uniform(-5, 5, (len(X), 2))
    return Y

def visualize2d(data, title, color=('#000000', '#D30000')):
    for id, sub in enumerate(data):
        plt.scatter(sub[:,0], sub[:,1], color=color[id])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

def visualize3d(data, title, color='#000000'):
    ax = plt.figure().add_subplot(projection='3d')
    for id, sub in enumerate(data):
        ax.scatter(sub[:,0], sub[:,1], sub[:,2], color=color[id])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()

def main():
    # generate data
    vis = True
    function_name = f'{STRUCTURE}_{N_DIM}d'
    X = np.linspace(-4, 4, N_TOT_POINTS)
    X = FUNC_DICT[function_name](X)
    np.random.shuffle(X)
    X_train, X_test = X[:int(TRAIN_TEST_SPLIT*N_TOT_POINTS)], X[int(TRAIN_TEST_SPLIT*N_TOT_POINTS):]
    if vis:
        visualize2d((X_train, X_test), function_name, color=('b', 'r'))
    np.savez(f'data/{function_name}_{N_POINTS}_points.npz', train_data=X_train, test_data=X_test)

FUNC_DICT = {
    "sin_exp_pol_3d": func1,
    "sin_pol_pol_3d": func2,
    "helix": func3,
    "plane": func4,
    "sin": func5,
    "circle": func6,
    "sin_2d": func7,
    "spiral": func8,
    'balls': func9,
    'random_2d': func10,
    }

if __name__ == '__main__':
    main()