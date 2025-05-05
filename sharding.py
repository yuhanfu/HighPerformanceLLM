import jax
import jax.numpy as jnp
import numpy as np

A = jnp.ones((1024, 1024))

devices = np.array(jax.devices()).reshape(4, 2)
mesh = jax.sharding.Mesh(devices, ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x', 'y'))
sharded_A = jax.device_put(A, sharding)

jax.debug.visualize_array_sharding(sharded_A)

print(f"{sharded_A.shape=} {sharded_A.addressable_shards[0].data.shape=}")