"""
See test_multihost.py for the actual tests that call the functions here.
"""
import os
import sys
from typing import Callable

import jax.distributed
import numpy as onp
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_array_info import sharding_info, sharding_vis, print_array_stats, simple_array_info
from test_utils import set_process_title

os.environ.pop('XLA_FLAGS', None)  # remove any flags that are inherited from the pytest parent

expected_failure_returncode = 55

test_name = sys.argv[1]
pindex = int(sys.argv[2])
num_processes = int(sys.argv[3])

set_process_title(f"jax-testrunner: {pindex}/{num_processes} {test_name}")

jax.distributed.initialize(
    coordinator_address="localhost:1299",
    num_processes=num_processes,
    process_id=pindex,
    local_device_ids=0,
    initialization_timeout=2  # only try to connect for 2s
)
devices = mesh_utils.create_device_mesh((num_processes,))
mesh = Mesh(devices, axis_names=('gpus',))
simple_sharding = NamedSharding(mesh, P(None, "gpus"))


def get_sharded_array(shape):
    return jax.numpy.zeros(shape)


get_sharded_array = jax.jit(
    get_sharded_array,
    out_shardings=simple_sharding,
    static_argnums=(0,)
)


# ---------- Start of Tests --------------

def run_empty():
    assert len(jax.devices()) == num_processes
    assert len(jax.local_devices()) == 1
    print(f"success at {jax.process_index()}")


def run_multihost_device_put():
    """
    basic array sharded along multiple hosts
    """
    arr = jax.numpy.zeros(shape=(32, 32, 32), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, simple_sharding)
    sharding_info(arr)
    sharding_vis(arr)
    print_array_stats()


def run_multihost_local_shard():
    """
    basic array sharded along multiple hosts
    """
    arr = jax.numpy.zeros(shape=(32, 32, 32), dtype=jax.numpy.complex64)
    arr: jax.Array = jax.device_put(arr, simple_sharding)
    # when sharding over multiple hosts, the array is not fully_adressable
    assert not arr.is_fully_addressable
    try:
        local_arr = onp.asarray(arr)
    except RuntimeError as e:
        # this means that they can't be read directly anymore
        print(e.args[0])
    # one can instead access the local shard as a normal jax array
    # it only contains a subset along the sharded axis
    local_subset = arr.addressable_data(0)
    local_subset_np = onp.asarray(local_subset)
    sharding_info(local_subset_np, "local_subset_np")


def run_multihost_closure():
    """
    Same as test_sharded_closure
    This does not work on multihost
    """
    arr = jax.numpy.zeros((16, 16))
    arr = jax.device_put(arr, NamedSharding(mesh, P("gpus")))

    def some_function():
        return arr * 5

    some_function = jax.jit(some_function, out_shardings=simple_sharding)
    try:
        out = some_function()
    except RuntimeError as e:
        print(e.args[0])
        exit(expected_failure_returncode)


def host_subset(array: onp.ndarray, size: int):
    """
    for now hard-coded to shard along axis 1
    """
    num_gpus = jax.device_count()
    host_id = jax.process_index()
    start = host_id * size // num_gpus
    end = (host_id + 1) * size // num_gpus
    return array[:, start:end]


def distribute_array(arr: onp.ndarray, mesh: Mesh) -> jax.Array:
    global_shape = arr.shape
    local_subset = host_subset(arr, arr.shape[1])
    local_subset_arr = jax.device_put(local_subset)
    arr_sharded = jax.make_array_from_single_device_arrays(
        global_shape,
        NamedSharding(mesh, P(None, "gpus")),
        [local_subset_arr])
    return arr_sharded


def run_numpy_to_sharded_array():
    """
    If we have some data as a numpy array and we want to transfer it to a sharded array
    we can copy the local subset of the array to the local GPU and then build the global sharded array from this
    using `jax.make_array_from_single_device_arrays`
    """
    local_np_array = onp.zeros((128, 128))
    distributed_array = distribute_array(local_np_array, mesh)
    assert local_np_array.shape == distributed_array.shape
    sharding_info(distributed_array, "distributed_array")


def run_process_allgather():
    """
    Test using `process_allgather()`
    """
    arr = get_sharded_array((128, 128))
    assert not arr.is_fully_addressable
    arr_np = process_allgather(arr)
    assert isinstance(arr_np, onp.ndarray)
    sharding_info(arr_np, "arr_np")
    assert arr_np.shape == arr.shape


def run_shard_map():
    arr = get_sharded_array((128, 128))

    def shard_func(x: jax.Array):
        simple_array_info(x, "x (in shard_map)")
        return jax.lax.psum(x, "gpus")

    func = shard_map(
        shard_func,
        mesh=mesh,
        in_specs=P(None, 'gpus'),
        out_specs=P(None),
    )
    out = func(arr)
    assert out.shape[1] == arr.shape[1] // num_processes
    sharding_info(out, "out")


if __name__ == '__main__':
    test_functions = {}
    f: Callable[[], None]
    for n, f in list(globals().items()):
        if n.startswith("run_"):
            test_functions[n] = f
    try:
        func = test_functions[test_name]
    except KeyError:
        raise RuntimeError(f"Unknown function name '{test_name}'")
    func()
