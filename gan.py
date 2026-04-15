import tensorflow as tf
import numpy as np

# data
real = np.array([[1,0,1,0],[0,1,0,1]], dtype=float)

# generator (2 → 4)
G = tf.keras.Sequential([tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(2,))])

# discriminator (4 → 1)
D = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(4,))])

D.compile(optimizer='adam', loss='binary_crossentropy')

# GAN (stack)
D.trainable = False
gan = tf.keras.Sequential([G, D])
gan.compile(optimizer='adam', loss='binary_crossentropy')


# training (minimal)
for i in range(10):

    # train D
    x_real = real[np.random.randint(0,2,1)]
    x_fake = G.predict(np.random.randn(1,2), verbose=0)
    
    X = np.vstack([x_real, x_fake])
    y = np.array([1,0])
    
    D.trainable = True
    D.fit(X, y, verbose=0)

    # train G
    D.trainable = False
    gan.fit(np.random.randn(1,2), np.array([1]), verbose=0)

    print("epoch", i)


# generate
print("\nGenerated + discriminator prediction:")
for _ in range(3):
    sample = G.predict(np.random.randn(1,2), verbose=0)
    disc_prob = D.predict(sample, verbose=0)[0][0]
    print("sample:", sample.round(), "| D(fake) probability:", round(float(disc_prob), 4))

print("\nDiscriminator prediction on real data:")
for x_real in real:
    real_prob = D.predict(x_real.reshape(1, 4), verbose=0)[0][0]
    print("real:", x_real, "| D(real) probability:", round(float(real_prob), 4))