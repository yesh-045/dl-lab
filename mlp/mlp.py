import random
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

# dataset
X = [[2,50],[4,60],[6,70],[8,80]]
Y = [0,0,1,1]

# normalize
X = [[x[0]/10, x[1]/100] for x in X]

# init weights
w_h = [[random.uniform(-1,1) for _ in range(2)] for _ in range(3)]
b_h = [random.uniform(-1,1) for _ in range(3)]

w_o = [random.uniform(-1,1) for _ in range(3)]
b_o = random.uniform(-1,1)

lr = 0.1

def forward(x):
    hidden = []
    
    for i in range(3):
        net = w_h[i][0]*x[0] + w_h[i][1]*x[1] + b_h[i]
        hidden.append(sigmoid(net))
    
    net_o = sum(w_o[i]*hidden[i] for i in range(3)) + b_o
    output = sigmoid(net_o)
    
    return hidden, output

def backward(x, hidden, output, d):
    global w_h, w_o, b_h, b_o
    
    # output error
    e_o = d - output
    delta_o = e_o * output * (1 - output)
    
    # hidden error
    delta_h = []
    for i in range(3):
        e_h = delta_o * w_o[i]
        delta_h.append(e_h * hidden[i]*(1-hidden[i]))
    
    # update output weights
    for i in range(3):
        w_o[i] += lr * delta_o * hidden[i]
    b_o += lr * delta_o
    
    # update hidden weights
    for i in range(3):
        w_h[i][0] += lr * delta_h[i] * x[0]
        w_h[i][1] += lr * delta_h[i] * x[1]
        b_h[i] += lr * delta_h[i]

# train
for epoch in range(50):
    for i in range(4):
        h, o = forward(X[i])
        backward(X[i], h, o, Y[i])

# accuracy
correct = 0
for i in range(4):
    _, o = forward(X[i])
    pred = 1 if o > 0.5 else 0
    if pred == Y[i]:
        correct += 1

print("Accuracy:", correct/4)

# test
test = [5/10, 65/100]
_, out = forward(test)
print("Prediction:", out)