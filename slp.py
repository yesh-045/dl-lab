import random

# dataset
X = [500, 800, 1000, 1200, 1500]
Y = [150, 220, 300, 360, 450]

# normalize
X = [x/1000 for x in X]
Y = [y/1000 for y in Y]

# init
w = random.uniform(-1,1)
b = random.uniform(-1,1)
lr = 0.01

def forward(x):
    return w*x + b

def train(x, y):
    global w, b
    
    y_pred = forward(x)
    error = y - y_pred
    
    # gradients
    dw = -2 * x * error
    db = -2 * error
    
    # update
    w -= lr * dw
    b -= lr * db

# train
for epoch in range(1000):
    for i in range(len(X)):
        train(X[i], Y[i])

# predict
x_test = 1100/1000
print("Prediction:", forward(x_test)*1000)