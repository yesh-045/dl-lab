import random

X_pattern = [
    [0,1,0,0],
    [1,1,1,0],
    [0,1,0,0],
    [0,0,0,0]
]

Y_pattern = 1

# 2x2 kernel (learnable)
K = [[random.uniform(-1,1) for _ in range(2)] for _ in range(2)]
lr = 0.01

def relu(x):
    return max(0, x)

def conv2d_2x2(img, kernel):
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            s = 0
            for ki in range(2):
                for kj in range(2):
                    s += img[i+ki][j+kj] * kernel[ki][kj]
            row.append(s)
        out.append(row)
    return out

def forward_pattern(img):
    c = conv2d_2x2(img, K)
    total = 0
    for i in range(3):
        for j in range(3):
            total += relu(c[i][j])
    return total, c

def train_pattern():
    global K
    out, _ = forward_pattern(X_pattern)
    error = Y_pattern - out

    for i in range(2):
        for j in range(2):
            grad = 0
            for x in range(3):
                for y in range(3):
                    grad += X_pattern[x+i][y+j]
            K[i][j] += lr * error * grad

# train
for epoch in range(10):
    train_pattern()

print("Pattern Prediction:", forward_pattern(X_pattern)[0])