"""
check test_jax.py::test_custom_rfftn_sharded for more information
"""

from typing import Callable

import jax.distributed
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P


def fft_partitioner(fft_func: Callable[[jax.Array], jax.Array], partition_spec: P, sharding_rule=None):
    @custom_partitioning
    def func(x):
        return fft_func(x)

    def supported_sharding(sharding, shape):
        assert sharding is not None
        return NamedSharding(sharding.mesh, partition_spec)

    def partition(mesh, arg_shapes, result_shape):
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return mesh, fft_func, supported_sharding(arg_shardings[0], arg_shapes[0]), (
            supported_sharding(arg_shardings[0], arg_shapes[0]),)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    func.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=sharding_rule
    )
    return func


def _fftn_XY(x):
    return jax.numpy.fft.fftn(x, axes=[0, 1])


def _rfft_Z(x):
    return jax.numpy.fft.rfft(x, axis=2)


def _ifftn_XY(x):
    return jax.numpy.fft.ifftn(x, axes=[0, 1])


def _irfft_Z(x):
    return jax.numpy.fft.irfft(x, axis=2)


fftn_XY = fft_partitioner(_fftn_XY, P(None, None, "gpus"), sharding_rule="x y z -> x y z")
rfft_Z = fft_partitioner(_rfft_Z, P(None, "gpus"), sharding_rule="x y z -> x y z_new")
ifftn_XY = fft_partitioner(_ifftn_XY, P(None, None, "gpus"))
irfft_Z = fft_partitioner(_irfft_Z, P(None, "gpus"))


def _rfftn(x):
    x = rfft_Z(x)
    x = fftn_XY(x)
    return x


def _irfftn(x):
    x = ifftn_XY(x)
    x = irfft_Z(x)
    return x
