import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Convert sample
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
all_data = np.concatenate([x_train, x_test], axis=0)  # ~70000 δείγματα

# Center
mean_vec = np.mean(all_data, axis=0, keepdims=True)
centered_data = all_data - mean_vec

# Create dataset
batch_size = 250
dataset = tf.data.Dataset.from_tensor_slices((centered_data, centered_data))
dataset = dataset.shuffle(buffer_size=centered_data.shape[0]).batch(batch_size)


# Coder
class DenseNoBias(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()

        self.W = self.add_weight(
            shape=(in_dim, out_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='EncoderWeights'
        )
        self.activation = activation

    def call(self, inputs):

        z = tf.matmul(inputs, self.W)
        if self.activation == 'relu':
            return tf.nn.relu(z)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(z)
        else:
            return z


# Decoder
class TransposeDenseNoBias(tf.keras.layers.Layer):
    def __init__(self, tied_layer, activation=None):
        super().__init__()
        self.tied_layer = tied_layer
        self.activation = activation

    def call(self, inputs):

        W_enc = self.tied_layer.W
        W_t = tf.transpose(W_enc)
        z = tf.matmul(inputs, W_t)
        if self.activation == 'relu':
            return tf.nn.relu(z)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(z)
        else:
            return z


# Create model

class TiedAutoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_layer1 = DenseNoBias(in_dim=784, out_dim=256, activation='relu')
        self.enc_layer2 = DenseNoBias(in_dim=256, out_dim=128, activation='relu')
        # Decoder (tied)
        self.dec_layer1 = TransposeDenseNoBias(tied_layer=self.enc_layer2, activation='relu')
        self.dec_layer2 = TransposeDenseNoBias(tied_layer=self.enc_layer1, activation='sigmoid')

    def call(self, inputs):
        # Forward encoder
        e1 = self.enc_layer1(inputs)
        e2 = self.enc_layer2(e1)
        # Forward decoder με μεταφορά των ίδιων βαρών
        d1 = self.dec_layer1(e2)
        d2 = self.dec_layer2(d1)
        return d2


# Train

model = TiedAutoencoder()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy'
)

epochs = 40
history = model.fit(dataset, epochs=epochs, verbose=1)

nparams = np.sum([np.prod(v.shape) for v in model.trainable_variables])
print("Συνολικές trainable παράμετροι (tied AE):", nparams)
