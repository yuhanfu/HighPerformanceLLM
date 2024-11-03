import jax
import jax.numpy as jnp
import timing_util

MATRIX_DIM = 32768


def f(A, B, C):
    return A + B + C


A = jnp.ones((MATRIX_DIM, MATRIX_DIM))
B = jnp.ones((MATRIX_DIM, MATRIX_DIM))
C = jnp.ones((MATRIX_DIM, MATRIX_DIM))

average_time_sec_no_jit = timing_util.simple_timeit(f, A, B, C, task="f") / 1000
print(f"average time for f: {average_time_sec_no_jit}")

jit_f = jax.jit(f)
average_time_sec_jit = timing_util.simple_timeit(jit_f, A, B, C, task="jit_f") / 1000
print(f"average time for jit_f: {average_time_sec_jit}")


def f2(A, B):
    return jax.nn.relu(A @ B)


num_bytes = A.size * 4
total_num_bytes_crossing_to_hbm = num_bytes * 3
total_num_flops = 2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM + MATRIX_DIM * MATRIX_DIM

average_time_sec_no_jit = timing_util.simple_timeit(f2, A, B, task="f2") / 1000

print(f"arithmetic intensity: {total_num_flops/total_num_bytes_crossing_to_hbm}")
print(f"average time for f2: {average_time_sec_no_jit}")
print(f"tera flops per second: {total_num_flops/average_time_sec_no_jit/10**12}")
print(
    f"gigabytes per second: {total_num_bytes_crossing_to_hbm/average_time_sec_no_jit/10**9}"
)

jit_f2 = jax.jit(f2)
average_time_sec_jit = timing_util.simple_timeit(f2, A, B, task="jit_f2") / 1000
print(f"average time for jit_f2: {average_time_sec_jit}")
print(f"tera flops per second: {total_num_flops/average_time_sec_jit/10**12}")
print(
    f"gigabytes per second: {total_num_bytes_crossing_to_hbm/average_time_sec_jit/10**9}"
)
