import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE


def multi_head_attention(X, num_heads=2):
    d_model = X.shape[1]
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    
    outputs = []
    
    for _ in range(num_heads):
        Wq = np.random.randn(d_model, head_dim)
        Wk = np.random.randn(d_model, head_dim)
        Wv = np.random.randn(d_model, head_dim)
        
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        
        out, _ = attention(Q, K, V)
        outputs.append(out)
    
    return np.concatenate(outputs, axis=1)

def layer_norm(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-6)

def feed_forward(x):
    W1 = np.random.randn(x.shape[1], x.shape[1]*2)
    W2 = np.random.randn(x.shape[1]*2, x.shape[1])
    return np.maximum(0, x @ W1) @ W2  # ReLU

def transformer_block(X, num_heads=2):
    attn = multi_head_attention(X, num_heads=num_heads)
    X = layer_norm(X + attn)
    
    ff = feed_forward(X)
    X = layer_norm(X + ff)
    
    return X


if __name__ == "__main__":
    X = np.array([[1, 0, 1, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 1]], dtype=float)

    PE = positional_encoding(3, 4)
    X = X + PE

    out = transformer_block(X, num_heads=2)
    print(out)


