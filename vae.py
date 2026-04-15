import tensorflow as tf
import numpy as np

# data (16-dim)
x = np.array([
 [1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1],
 [0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0]
], dtype=float)

# encoder
inp = tf.keras.Input(shape=(16,))
h = tf.keras.layers.Dense(4, activation='relu')(inp)
mu = tf.keras.layers.Dense(2)(h)
logvar = tf.keras.layers.Dense(2)(h)

# sample
z = mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(mu))

# decoder
out = tf.keras.layers.Dense(16, activation='sigmoid')(z)

# model
vae = tf.keras.Model(inp, out)

# loss
loss = tf.reduce_mean((inp - out)**2) + \
       -0.5 * tf.reduce_mean(1 + logvar - mu**2 - tf.exp(logvar))

vae.add_loss(loss)
vae.compile(optimizer='adam')

# train
vae.fit(x, x, epochs=10, verbose=0)

# generate
z_sample = np.random.randn(1,2)
print(tf.keras.layers.Dense(16, activation='sigmoid')(z_sample).numpy().round())