# external imports
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, Model, optimizers, initializers

# internal imports
from config import *

class Autoencoder:
    def __init__(self, input_dim, encoding_dim, hidden_layers=[], activation='tanh', optimizer='adam', learning_rate=0.01):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        
        self.learning_rate = learning_rate
        self.initializer = initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=132) # seed 5 before
        self.activation = activation
        loaded_optimizers = {
        'adam': optimizers.Adam(learning_rate=self.learning_rate),
        'sgd': optimizers.SGD(learning_rate=self.learning_rate),
        }
        self.optimizer_obj = loaded_optimizers[optimizer]

        self._autoencoder = self._build_autoencoder()

    def _build_autoencoder(self):
        input_layer = layers.Input(shape=(self.input_dim,))

        # Encoder layers
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = layers.Dense(units, 
                                   activation=self.activation,
                                   kernel_initializer=self.initializer,
                                   bias_initializer='zeros')(encoded)

        # Bottleneck layer
        encoded = layers.Dense(self.encoding_dim, 
                               activation=self.activation, 
                               kernel_initializer=self.initializer,
                               bias_initializer='zeros')(encoded)

        # Decoder layers
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = layers.Dense(units, 
                                   activation=self.activation,
                                   kernel_initializer=self.initializer,
                                   bias_initializer='zeros')(decoded)

        # Output layer
        decoded = layers.Dense(self.input_dim,
                               activation=None, # output activation
                               kernel_initializer=self.initializer,
                               bias_initializer='zeros')(decoded)

        # Models
        self._encoder = Model(inputs=input_layer,
                        outputs=encoded,
                        name='encoder',
                        )
        self._decoder = Model(inputs=encoded,
                        outputs=decoded,
                        name='decoder',
                        )
        autoencoder = Model(inputs=input_layer, 
                                   outputs=decoded, 
                                   name='autoencoder',
                                   )

        # Compile the model
        autoencoder.compile(optimizer=self.optimizer_obj,
                            loss='mse',
                            metrics=[tf.keras.metrics.RootMeanSquaredError()],
                            )

        return autoencoder

    def train(self, x_train, epochs=10, batch_size=32, save_freq = 'epoch', model_path = 'autoencoder_model'):
        checkpoint_path = model_path+'/cp-{epoch:03d}-{batch:03d}.weights.h5'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
         os.makedirs(checkpoint_dir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=False, 
                    verbose=1, 
                    save_weights_only=True,
                    save_freq=save_freq,              # integer for save after number of batches or "epoch" after each epoch
                    )
        self._autoencoder.fit(x_train, x_train, 
                            epochs=epochs,
                            verbose='auto', 
                            batch_size=batch_size, 
                            shuffle=True,
                            callbacks=[cp_callback],
                            )

    def encode(self, x):
        return self._encoder.predict(x)

    def decode(self, encoded_data):
        return self._decoder.predict(encoded_data)
    
    def predict(self, x, **kwargs):
        return self._autoencoder.predict(x, **kwargs)


# Example usage
if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(42)
    tf.random.set_seed(42)

    arr = np.load(DATA_PATH)
    x_train = arr['train_data']
    x_test = arr['test_data']
    print(x_train.shape)
    print(x_test.shape)

    # Create and train the autoencoder
    autoencoder = Autoencoder(input_dim=N_DIM, 
                              encoding_dim=ENCODED_DIM, 
                              hidden_layers=HIDDEN_LAYERS, 
                              optimizer=OPTIMIZER, 
                              learning_rate=LEARNING_RATE,
                              )
    autoencoder._autoencoder.summary()
    autoencoder.train(x_train,
                      epochs=N_EPOCHS, 
                      batch_size=BATCH_SIZE,
                      save_freq=1,
                      model_path=MODEL_PATH,
                      )

    # Encode and decode some data
    encoded_data = autoencoder.encode(x_test)
    decoded_data = autoencoder.decode(encoded_data)

    # Evaluate reconstruction error
    mse = np.mean(np.square(x_test - decoded_data))
    print("Mean Squared Error (MSE) on Test Data:", mse)