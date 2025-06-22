import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import numpy as np
import optax

import time

import functools

import orbax.checkpoint as ocp

import os

from jax.experimental.pallas.ops.tpu import flash_attention as pallas_attention

BATCH_SIZE = 1
SEQUENCE_LEN = 128

VOCAB_DIM = 256
EMBED_DIM = 1024
FF_DIM = 4096

NUM_HEAD = 4
HEAD_DIM = 128

LAYERS = 8

LEARNING_RATE = 1e-6

FSDP = 1
TENSOR = 1


mesh = jax.sharding.Mesh(np.array(jax.devices()[0]).reshape(FSDP, TENSOR), ("fsdp", "tp"))
desired_embedding_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("fsdp", None, "tp")
)  # apply to BATCH, SEQUENCE, EMBED
home_dir = os.environ["HOME"]


def attention_with_masking(Q, K, V, seq_pos):
    query_segment_id = jnp.ones((1, Q.shape[1]), dtype=jnp.int32)
    kv_segment_id = jnp.ones((1, SEQUENCE_LEN), dtype=jnp.int32) * jnp.expand_dims(
        jnp.arange(SEQUENCE_LEN) <= seq_pos, axis = 0
    )

    segment_ids = pallas_attention.SegmentIds(q=query_segment_id, kv=kv_segment_id)
    return jnp.swapaxes(
        pallas_attention.mha_reference(
            jnp.swapaxes(Q, 1, 2),
            jnp.swapaxes(K, 1, 2),
            jnp.swapaxes(V, 1, 2),
            None,
            segment_ids=segment_ids
        ),
        1,
        2
    )


class Model(nn.Module):
    @nn.compact
    def __call__(self, x, pos, kv_cache):
        """inputs is [BATCH_SIZE, SEQUENCE_LEN]"""
        embedding = self.param(
            "embedding",
            nn.with_partitioning(nn.initializers.normal(1), ("tp", "fsdp")),
            (VOCAB_DIM, EMBED_DIM),
            jnp.float32,
        )
        x = jnp.asarray(embedding)[x]  # output should be [BATCH, SEQUENCE, EMBED]

        for i in range(LAYERS):
            x = nn.LayerNorm(name="layer_norm_" + str(i))(x)
           
            positional_embedding = self.param(
                "positional_embedding_" + str(i),
                nn.with_partitioning(nn.initializers.normal(1), (None, None, "fsdp")),
                (1, SEQUENCE_LEN, EMBED_DIM),
                jnp.float32,
            )
            x += jax.lax.dynamic_slice_in_dim(positional_embedding, pos, 1, axis=1)
            x = jax.lax.with_sharding_constraint(x, desired_embedding_sharding)

            feedforward = self.param(
                "feedforward_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
                (EMBED_DIM, FF_DIM),
                jnp.float32,
            )
            x = x @ feedforward
            x = jax.nn.relu(x)
            x = jax.lax.with_sharding_constraint(x, desired_embedding_sharding)
            embed = self.param(
                "embed_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("tp", "fsdp")),
                (FF_DIM, EMBED_DIM),
                jnp.float32,
            )
            x = x @ embed
            x = jax.nn.relu(x)

            q_proj = self.param(
                "qproj_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
                (EMBED_DIM, NUM_HEAD, HEAD_DIM),
                jnp.float32,
            )
            q = jnp.einsum("BSE,EHD->BSHD", x, q_proj)
            
            k_proj = self.param(
                "kproj_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
                (EMBED_DIM, NUM_HEAD, HEAD_DIM),
                jnp.float32,
            )
            k = jnp.einsum("BSE,EHD->BSHD", x, k_proj)
            kv_cache[f"key_{i}"] = jax.lax.dynamic_update_index_in_dim(kv_cache[f"key_{i}"], k, pos, 1)
            k = kv_cache[f"key_{i}"]

            v_proj = self.param(
                "vproj_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
                (EMBED_DIM, NUM_HEAD, HEAD_DIM),
                jnp.float32,
            )
            v = jnp.einsum("BSE,EHD->BSHD", x, v_proj)
            kv_cache[f"value_{i}"] = jax.lax.dynamic_update_index_in_dim(kv_cache[f"value_{i}"], v, pos, 1)
            v = kv_cache[f"value_{i}"]

            o = attention_with_masking(q, k, v, pos)
            o_proj = self.param(
                "oproj_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
                (NUM_HEAD, HEAD_DIM, EMBED_DIM),
                jnp.float32,
            )
            x = jnp.einsum("BSHD,HDE->BSE", o, o_proj)
            x = jax.lax.with_sharding_constraint(x, desired_embedding_sharding)

        return x @ jnp.asarray(embedding).T


def convert_to_ascii(string_array, max_length):
    result = np.zeros((len(string_array), max_length), dtype=np.uint8)
    for i, string in enumerate(string_array):
        for j, char in enumerate(string):
            if j >= max_length:
                break
            result[i, j] = char
    return result


def input_to_output(np_array):
    zero_array = np.zeros((BATCH_SIZE, SEQUENCE_LEN), dtype=jnp.uint8)
    zero_array[:, 1:SEQUENCE_LEN] = np_array[:, 0 : SEQUENCE_LEN - 1]
    return zero_array


def calculate_loss(params, model, inputs, outputs):
    proposed_outputs = model.apply(params, inputs)
    one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
    loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
    return jnp.mean(loss)


def step(state, model, inputs, outputs):
    loss, grad = jax.value_and_grad(calculate_loss)(
        state.params, model, inputs, outputs
    )
    state = state.apply_gradients(grads=grad)
    return loss, state


def calculate_num_params(pytree):
    sizes = jax.tree_util.tree_map(lambda x: x.size, pytree)
    return jax.tree_util.tree_reduce(lambda x, y: x + y, sizes)


def numpy_to_string(numpy_arr):
    return "".join([chr(item) for item in numpy_arr])


def create_kv_cache():
    output = {}
    for i in range(LAYERS):
        output[f"key_{i}"] = jnp.zeros((BATCH_SIZE, SEQUENCE_LEN, NUM_HEAD, HEAD_DIM))
        output[f"value_{i}"] = jnp.zeros((BATCH_SIZE, SEQUENCE_LEN, NUM_HEAD, HEAD_DIM))
    return output


def main():
    ds = tfds.load("lm1b", split="train", shuffle_files=False)
    ds = ds.batch(BATCH_SIZE)

    kv_cache = create_kv_cache()

    rngkey = jax.random.key(0)
    model = Model()

    shape_init = jax.eval_shape(
        functools.partial(model.init, rngkey),
        jax.ShapeDtypeStruct((BATCH_SIZE, 1), dtype=jnp.uint8),
        0,
        kv_cache,
    )
    state_sharding = nn.get_sharding(shape_init, mesh)
    init_params = jax.jit(model.init, out_shardings=state_sharding)(
        rngkey, 
        jax.ShapeDtypeStruct((BATCH_SIZE, 1), dtype=jnp.uint8), 
        0, 
        kv_cache,
    )

    num_total_floats = calculate_num_params(init_params) + calculate_num_params(kv_cache)
    num_bytes_to_read = num_total_floats * 4

    print(f"Number bytes {num_bytes_to_read/1e9} GB")

    tx = optax.adam(learning_rate=LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=tx
    )

    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(
        f"{home_dir}/HighPerformanceLLM/checkpoint/checkpoint_0078000", 
        abstract_state,
    )

    text = np.zeros((1, 1), dtype = np.int32)
    NUM_TOKENS = 30
    output_string = ""
    for i in range(NUM_TOKENS):
        logits = model.apply(state.params, text, i, kv_cache)
        text = jnp.argmax(logits, axis=2)
        output_string += chr(text[0, 0])

    print(f"Output: `{output_string}`")


if __name__ == "__main__":
    main()
