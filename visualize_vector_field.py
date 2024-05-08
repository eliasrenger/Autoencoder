import matplotlib.pyplot as plt
import numpy as np

def func(X):
    # define the function
    Y = X.copy()
    Y[:, 0] = X[:, 1]
    Y[:, 1] = -X[:, 0]
    return Y

def vis_vec_field(vec, point, title = 'Vector field'):
    color = 'r'
    #max_err = np.max(np.abs(ae_dynamics), axis=0)
    #plot_border = grid_intervals + vector_scale * np.array([-max_err, max_err]).T
    #plot_border[:,0], plot_border[:,1] = np.floor(plot_border[:,0]), np.ceil(plot_border[:,1])

    # plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(point[:,0], point[:,1], point[:,2], 
            vec[:, 0], vec[:, 1], vec[:, 2], 
                normalize=True, color=color)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #ax.set_xlim(plot_border[0,0], plot_border[0,1])
    #ax.set_ylim(plot_border[1,0], plot_border[1,1])
    #ax.set_zlim(plot_border[2,0], plot_border[2,1])
    plt.show()

base = np.linspace(-3, 3, 10)
X = np.array(np.meshgrid(base, base, base)).T.reshape(-1, 3)
Y = func(X)
vec = Y - X
vis_vec_field(vec, X, 'Vector field')