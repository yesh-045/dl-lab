import numpy as np

# image rows as sequence
X = np.array([
    [1,1,1],
    [0,1,0],
    [1,1,1]
], dtype=float)

# target: arbitrary class (e.g., 1 = pattern exists)
Y = np.array([[1]])


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - x**2

input_size = 3
hidden_size = 4
output_size = 1

Wx = np.random.randn(input_size, hidden_size)
Wh = np.random.randn(hidden_size, hidden_size)
Wy = np.random.randn(hidden_size, output_size)

bh = np.zeros((1, hidden_size))
by = np.zeros((1, output_size))

def forward(X):
    h = np.zeros((1, hidden_size))
    hs = []
    
    for t in range(len(X)):
        x_t = X[t].reshape(1, -1)
        h = tanh(x_t @ Wx + h @ Wh + bh)
        hs.append(h)
    
    y = h @ Wy + by
    return y, hs


def train(lr, epochs):
    global Wx, Wh, Wy, bh, by
    
    for epoch in range(epochs):
        # forward
        y_pred, hs = forward(X)
        loss = np.mean((Y - y_pred)**2)
        
        # output gradient
        dy = (y_pred - Y)
        dWy = hs[-1].T @ dy
        dby = dy
        
        dh_next = dy @ Wy.T
        
        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        dbh = np.zeros_like(bh)
        
        # backward through time
        for t in reversed(range(len(X))):
            h = hs[t]
            dh = dh_next * tanh_deriv(h)
            
            x_t = X[t].reshape(1, -1)
            h_prev = hs[t-1] if t > 0 else np.zeros_like(h)
            
            dWx += x_t.T @ dh
            dWh += h_prev.T @ dh
            dbh += dh
            
            dh_next = dh @ Wh.T
        
        # update
        Wx -= lr * dWx
        Wh -= lr * dWh
        Wy -= lr * dWy
        bh -= lr * dbh
        by -= lr * dby
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

train(0.01, 10)
train(0.1, 10)

y_pred, _ = forward(X)
print("Prediction:", y_pred)



'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

# Optional reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# image rows as sequence: (samples, timesteps, features)
X = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1]
], dtype=np.float32)
X = X.reshape(1, 3, 3)   # 1 sample, 3 time steps, 3 features

# target
Y = np.array([[1]], dtype=np.float32)

# Built-in RNN model
model = Sequential([
    SimpleRNN(4, activation="tanh", input_shape=(3, 3)),
    Dense(1)  # linear output
])

# Train with lr=0.01
model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
model.fit(X, Y, epochs=10, verbose=0)

# Train with lr=0.1 (re-compile with new optimizer)
model.compile(optimizer=Adam(learning_rate=0.1), loss="mse")
model.fit(X, Y, epochs=10, verbose=0)

# Prediction
y_pred = model.predict(X, verbose=0)
print("Prediction:", y_pred)

'''