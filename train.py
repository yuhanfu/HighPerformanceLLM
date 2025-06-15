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

BATCH_SIZE = 384
SEQUENCE_LEN = 128

VOCAB_DIM = 256
EMBED_DIM = 1024
FF_DIM = 4096

NUM_HEAD = 4
HEAD_DIM = 128

LAYERS = 8

LEARNING_RATE = 1e-6

FSDP = 4
TENSOR = 1

LOG_PERIOD = 10
CHECKPOINT_PERIOD = 1000

mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(FSDP, TENSOR), ("fsdp", "tp"))
desired_embedding_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("fsdp", None, "tp")
)  # apply to BATCH, SEQUENCE, EMBED
home_dir = os.environ["HOME"]


def attention(Q, K, V):
    weights_unnormalized = jnp.einsum("bshd,bthd->bhst", Q, K)
    weights_unnormalized_to_zero_out = jnp.triu(
        jnp.ones((SEQUENCE_LEN, SEQUENCE_LEN), dtype=jnp.bfloat16), k=1
    )
    weights = jax.nn.softmax(
        weights_unnormalized - 1e6 * weights_unnormalized_to_zero_out
    )
    output = jnp.einsum("bhst,bthd->bshd", weights, V)
    return output


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
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
            x += positional_embedding
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
            v_proj = self.param(
                "vproj_" + str(i),
                nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
                (EMBED_DIM, NUM_HEAD, HEAD_DIM),
                jnp.float32,
            )
            v = jnp.einsum("BSE,EHD->BSHD", x, v_proj)

            o = attention(q, k, v)
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


def main():
    ds = tfds.load("lm1b", split="train", shuffle_files=False)
    ds = ds.batch(BATCH_SIZE)

    rngkey = jax.random.key(0)
    model = Model()

    shape_init = jax.eval_shape(
        functools.partial(model.init, rngkey),
        jax.ShapeDtypeStruct((BATCH_SIZE, SEQUENCE_LEN), dtype=jnp.uint8),
    )
    state_sharding = nn.get_sharding(shape_init, mesh)
    init_params = jax.jit(model.init, out_shardings=state_sharding)(
        rngkey, jax.ShapeDtypeStruct((BATCH_SIZE, SEQUENCE_LEN), dtype=jnp.uint8)
    )

    num_total_params = calculate_num_params(init_params)
    num_total_flops = 6 * BATCH_SIZE * SEQUENCE_LEN * num_total_params
    num_total_flops_per_device = num_total_flops / jax.device_count()

    tx = optax.adam(learning_rate=LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=tx
    )

    static_step = jax.jit(step, static_argnums=1)
    stepnum = 0

    last_step_time = time.time()
    for example in ds:
        outputs = convert_to_ascii(example["text"].numpy(), SEQUENCE_LEN)
        inputs = input_to_output(outputs)
        loss, state = static_step(state, model, inputs, outputs)

        if stepnum % CHECKPOINT_PERIOD == 0:
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(f"{home_dir}/HighPerformanceLLM/checkpoint/checkpoint_{stepnum:07}", state)
            print(f"Saved checkpoint at {stepnum}")

        stepnum += 1
        # data overloading
        if stepnum % LOG_PERIOD == 0:
            new_time = time.time()
            time_elapsed_seconds = new_time - last_step_time
            last_step_time = new_time
            print(f"{stepnum} -> {loss} {time_elapsed_seconds}")
            per_device_tflops_completed_in_interval = (
                num_total_flops_per_device * LOG_PERIOD / 1e12
            )
            print(
                f"TFLOPS/s/device: {per_device_tflops_completed_in_interval / time_elapsed_seconds}"
            )


if __name__ == "__main__":
    main()
