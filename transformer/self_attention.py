import numpy as np
from transformer.transformer import transformer_block, positional_encoding, softmax

X = np.array([[1,0,1,0,1]], dtype=float)  # (1,5)

# project
Wq = np.random.randn(5,5)
Wk = np.random.randn(5,5)
Wv = np.random.randn(5,5)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.T / np.sqrt(5)
weights = softmax(scores)

out = weights @ V

print("Attention weights:", weights)
print("Output:", out)



#transformer on pixel sequence
X = np.array([[1,0,1,1,0]], dtype=float).T  # shape (5,1)

# expand dimension
X = np.hstack([X]*4)  # fake embedding dim = 4

PE = positional_encoding(5, 4)
X = X + PE

out = transformer_block(X)

# predict next value (simple linear head)
W_out = np.random.randn(4,1)
pred = out[-1] @ W_out

print("Next pixel prediction:", pred)