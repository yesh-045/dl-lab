import tensorflow as tf
import numpy as np

X = np.array([
    [[1,1,1],[0,1,0],[1,1,1]],  # pattern → 1
    [[0,0,0],[1,0,1],[0,0,0]]   # no pattern → 0
], dtype=float)

Y = np.array([1,0])

# model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(4, input_shape=(3,3)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train
model.fit(X, Y, epochs=10, verbose=0)

# predict
pred = model.predict(X, verbose=0)

print("Raw:", pred)
print("Class:", (pred > 0.5).astype(int))