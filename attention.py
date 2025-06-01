import jax
import jax.numpy as jnp

BATCH = 1
HEADS = 4
SEQUENCE = 2048
HEAD_DIM = 128

Q = jax.random.normal(jax.random.key(0), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
K = jax.random.normal(jax.random.key(1), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
V = jax.random.normal(jax.random.key(2), (BATCH, SEQUENCE, HEADS, HEAD_DIM))

def attention(Q, K, V):
  weights_unnormalized = jnp.einsum('bshd,bthd->bhst', Q, K)
  weights_unnormalized_to_zero_out = jnp.triu(jnp.ones((SEQUENCE, SEQUENCE), dtype=jnp.bfloat16), k=1)
  weights = jax.nn.softmax(weights_unnormalized - 1e6 * weights_unnormalized_to_zero_out)
  print(f"{weights[0, 0, 20, :]=}")
  output = jnp.einsum('bhst,bthd->bshd', weights, V)
  return output

att = attention(Q,K,V)