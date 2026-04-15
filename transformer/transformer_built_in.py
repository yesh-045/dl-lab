import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Optional: reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Input (batch, seq_len, d_model)
X = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
], dtype=np.float32)[None, :, :]   # shape: (1, 3, 3)

seq_len = X.shape[1]
d_model = X.shape[2]

x = tf.convert_to_tensor(X)

# Simple built-in positional information (learned position embedding)
pos = tf.range(start=0, limit=seq_len, delta=1)                 # (3,)
pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)(pos)  # (3, 3)
x = x + pos_emb[tf.newaxis, :, :]                                # (1, 3, 3)

# Transformer block using inbuilt layers
mha = layers.MultiHeadAttention(num_heads=1, key_dim=d_model)
attn_out = mha(query=x, value=x, key=x)                          # self-attention
x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)        # residual + norm

ffn = tf.keras.Sequential([
    layers.Dense(d_model * 2, activation="relu"),
    layers.Dense(d_model)
])
ff_out = ffn(x)
x = layers.LayerNormalization(epsilon=1e-6)(x + ff_out)          # residual + norm

print(x.numpy()[0])