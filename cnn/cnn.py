import random

X = [
 [0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [0,0,0,0]
]

Y = 1

# kernel 2x2
K = [[random.uniform(-1,1) for _ in range(2)] for _ in range(2)]
lr = 0.01

def relu(x):
    return max(0,x)

def conv(img):
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            s = 0
            for ki in range(2):
                for kj in range(2):
                    s += img[i+ki][j+kj]*K[ki][kj]
            row.append(s)
        out.append(row)
    return out

def forward(img):
    c = conv(img)
    s = 0
    for i in range(len(c)):
        for j in range(len(c[0])):
            s += relu(c[i][j])
    return s, c

def train():
    global K
    out, c = forward(X)
    error = Y - out
    
    for i in range(2):
        for j in range(2):
            grad = 0
            for x in range(3):
                for y in range(3):
                    grad += X[x+i][y+j]
            K[i][j] += lr * error * grad

# train
for epoch in range(10):
    train()

print("Prediction:", forward(X)[0])