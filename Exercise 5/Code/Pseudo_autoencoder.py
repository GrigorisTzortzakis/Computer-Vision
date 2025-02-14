import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.

all_data = np.concatenate([x_train, x_test], axis=0)
mean_vec = np.mean(all_data, axis=0, keepdims=True)
centered_data = all_data - mean_vec

batch_size = 250
dataset = tf.data.Dataset.from_tensor_slices((centered_data, centered_data))
dataset = dataset.shuffle(buffer_size=centered_data.shape[0]).batch(batch_size)


# Set no bias

class DenseNoBias(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, activation='relu'):
        super().__init__()
        self.activation = activation
        self.W = self.add_weight(
            shape=(in_dim, out_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='EncoderWeights'
        )

    def call(self, inputs):
        z = tf.matmul(inputs, self.W)
        if self.activation == 'relu':
            return tf.nn.relu(z)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(z)
        else:
            return z


# Pseudo decoder

class PseudoIsotropicDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, in_dim, activation='relu'):
        super().__init__()
        self.activation = activation

        init_mat = tf.eye(num_rows=in_dim, num_columns=out_dim, dtype=tf.float32)
        self.W_pseudo = tf.Variable(
            initial_value=init_mat,
            trainable=False,
            name='PseudoW'
        )

    def call(self, inputs):
        z = tf.matmul(inputs, self.W_pseudo)
        if self.activation == 'relu':
            return tf.nn.relu(z)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(z)
        else:
            return z


# Half parameters

class PseudoIsotropicAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DenseNoBias(in_dim=784, out_dim=256, activation='relu')
        self.enc2 = DenseNoBias(in_dim=256, out_dim=128, activation='relu')
        # Decoder
        self.dec1 = PseudoIsotropicDecoderLayer(out_dim=256, in_dim=128, activation='relu')
        self.dec2 = PseudoIsotropicDecoderLayer(out_dim=784, in_dim=256, activation='sigmoid')

    def call(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        d2 = self.dec2(d1)
        return d2


# Decoder
#

def initialize_pseudo_weights(model: PseudoIsotropicAE):
    s, U, V = tf.linalg.svd(model.enc2.W, full_matrices=False)
    W_pseudo2 = tf.matmul(V, tf.transpose(U))  # (128,256)
    model.dec1.W_pseudo.assign(W_pseudo2)

    s1, U1, V1 = tf.linalg.svd(model.enc1.W, full_matrices=False)
    W_pseudo1 = tf.matmul(V1, tf.transpose(U1))  # (256,784)
    model.dec2.W_pseudo.assign(W_pseudo1)

    print("Pseudo-isotropic initialization done via SVD.")


model = PseudoIsotropicAE()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy'
)

# forward
dummy_inp = tf.zeros((1, 784))
_ = model(dummy_inp)

initialize_pseudo_weights(model)

# Train

epochs = 40
model.fit(dataset, epochs=epochs, verbose=1)

nparams = np.sum([np.prod(v.shape) for v in model.trainable_variables])
print(f"Trainable Παράμετροι (pseudo-isotropic AE): {nparams}")
