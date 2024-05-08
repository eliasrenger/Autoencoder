import numpy as np
from autoencoder import Autoencoder
# load model
data_name = 'sin_2d_10000_points'
model_id = '0030'
model_path = f'models/{data_name}/cp-{model_id}.weights.h5'
# Load the model
model = Autoencoder(2, 1, hidden_layers=[2])
model._autoencoder.load_weights(model_path)

orbit = np.array([[3.83396856, 1.08431428],
       [4.38352533, 1.89752951],
       [3.71086577, 1.25866642],
       [3.71340171, 1.19039219],
       [3.92511143, 1.48235628],
       [3.77088165, 1.13668857],
       [3.78331329, 1.13077362],
       [3.71912396, 1.18053611],
       [3.73810682, 1.18546653]])

orbit_1 = np.roll(model.predict(orbit), 1, axis=0)
print(orbit_1)
diff = orbit - orbit_1
print(diff)
print(f'total difference: {np.linalg.norm(diff)}')