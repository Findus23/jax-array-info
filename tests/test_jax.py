import os
from functools import partial

import jax.numpy
import numpy as np
import pytest
from jax._src.config import use_shardy_partitioner
from jax._src.sharding_impls import PositionalSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_array_info import sharding_info, sharding_vis, print_array_stats, simple_array_info

from test_utils import is_on_cluster

num_gpus = 8

if is_on_cluster():
    jax.distributed.initialize()
else:
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_gpus}'

devices = mesh_utils.create_device_mesh((num_gpus,))
mesh = Mesh(devices, axis_names=('gpus',))

simple_sharding = NamedSharding(mesh, P(None, "gpus"))
simple_sharding1d = NamedSharding(mesh, P("gpus"))

devices_2d = mesh_utils.create_device_mesh((num_gpus // 2, 2))
mesh_2d = Mesh(devices_2d, axis_names=('a', 'b'))
devices_3d = mesh_utils.create_device_mesh((num_gpus // 4, 2, 2))
mesh_3d = Mesh(devices_3d, axis_names=('a', 'b', 'c'))


def generalize(input: str) -> str:
    """
    modify test output so that tests still succeed when running on multi-host GPU cluster
    """
    input = input.replace('GPU', "CPU")
    return "".join(l for l in input.splitlines(keepends=True) if "is_fully_addressable" not in l)

def test_simple(capsys):
    arr = jax.numpy.array([1, 2, 3])
    sharding_info(arr, "arr")
    sharding_vis(arr)

    assert generalize(capsys.readouterr().out) == """
╭──── arr ─────╮
│ shape: (3,)  │
│ dtype: int32 │
│ size: 12.0 B │
│ not sharded  │
╰──────────────╯
┌───────┐
│ CPU 0 │
└───────┘
""".lstrip()


def test_not_sharded(capsys):
    arr = jax.numpy.zeros(shape=(10, 10, 10), dtype=jax.numpy.complex64)
    sharding_info(arr)
    sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────╮
│ shape: (10, 10, 10) │
│ dtype: complex64    │
│ size: 7.8 KiB       │
│ not sharded         │
╰─────────────────────╯
───────────── showing dims [0, 1] from original shape (10, 10, 10) ─────────────
┌───────┐
│       │
│       │
│       │
│       │
│ CPU 0 │
│       │
│       │
│       │
│       │
└───────┘
""".lstrip()


def test_device_put(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, simple_sharding)
    sharding_info(arr)
    sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:4 (1/8) │
│                    Total size: 32           │
╰─────────────────────────────────────────────╯
───────────── showing dims [0, 1] from original shape (32, 32, 32) ─────────────
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
""".lstrip()


def test_operator_sharded(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, simple_sharding)
    arr = arr * 2
    sharding_info(arr)
    sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:4 (1/8) │
│                    Total size: 32           │
╰─────────────────────────────────────────────╯
───────────── showing dims [0, 1] from original shape (32, 32, 32) ─────────────
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
""".lstrip()


def test_jit_out_sharding_sharded(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)

    def func(x):
        return x * 2

    func = jax.jit(func, out_shardings=simple_sharding)
    arr = func(arr)
    sharding_info(arr)
    sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:4 (1/8) │
│                    Total size: 32           │
╰─────────────────────────────────────────────╯
───────────── showing dims [0, 1] from original shape (32, 32, 32) ─────────────
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
""".lstrip()


def test_positional_sharded(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, PositionalSharding(devices))
    sharding_info(arr)
    sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭───────────────────────────────────────────────────────────────────╮
│ shape: (32,)                                                      │
│ dtype: complex64                                                  │
│ size: 256.0 B                                                     │
│ PositionalSharding:                                               │
│ [{CPU 0} {CPU 1} {CPU 2} {CPU 3} {CPU 4} {CPU 5} {CPU 6} {CPU 7}] │
│ axis 0 is sharded: CPU 0 contains 0:4 (1/8)                       │
│                    Total size: 32                                 │
╰───────────────────────────────────────────────────────────────────╯
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
""".lstrip()


def test_in_jit(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, simple_sharding)

    def func(x):
        sharding_info(x)
        sharding_vis(x)
        return x * 2

    func = jax.jit(func)
    func(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ called in jit                               │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:4 (1/8) │
│                    Total size: 32           │
╰─────────────────────────────────────────────╯
───────────── showing dims [0, 1] from original shape (32, 32, 32) ─────────────
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │ CPU 4 │ CPU 5 │ CPU 6 │ CPU 7 │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
│       │       │       │       │       │       │       │       │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
""".lstrip()


def test_pmap(capsys):
    arr = jax.numpy.zeros(shape=(8, 8 * 3), dtype=jax.numpy.complex64)
    arr = jax.pmap(lambda x: x ** 2)(arr)
    sharding_info(arr)
    sharding_vis(arr)

    assert generalize(capsys.readouterr().out) == """
╭──────────────────────────────────────────────────────────────────────╮
│ shape: (8, 24)                                                       │
│ dtype: complex64                                                     │
│ size: 1.5 KiB                                                        │
│ PmapSharding(sharding_spec=ShardingSpec((Chunked(8), NoSharding()),  │
│ (ShardedAxis(axis=0),)), device_ids=[0, 1, 2, 3, 4, 5, 6, 7],        │
│ device_platform=CPU, device_shape=(8,))                              │
│ axis 0 is sharded: CPU 0 contains 0:1 (1/8)                          │
│                    Total size: 8                                     │
╰──────────────────────────────────────────────────────────────────────╯
Output for PmapSharding might be incorrect
┌─────────────────────────────────────────────────────────────────────────┐
│                                  CPU 0                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 1                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 2                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 3                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 4                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 5                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 6                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                  CPU 7                                  │
└─────────────────────────────────────────────────────────────────────────┘
""".lstrip()


def test_custom_rfftn_sharded(capsys):
    """
    more complex example of efficiently calculating an (i)rfftn
    of sharded arrays using custom_partitioning

    loosely based on the example from https://docs.jax.dev/en/latest/jax.experimental.custom_partitioning.html
    and xfft from https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFTMp/JAX_FFT
    """

    from fft_utils import _rfftn

    rfftn = jax.jit(
        _rfftn,
        in_shardings=simple_sharding,
        out_shardings=simple_sharding,
    )
    rfftn_original = jax.jit(
        jax.numpy.fft.rfftn,
        in_shardings=simple_sharding,
        out_shardings=simple_sharding
    )

    input_array = jax.numpy.zeros(shape=(128, 128, 128))
    input_array = jax.device_put(input_array, simple_sharding)
    sharding_info(input_array, "input_array")

    hlo = rfftn.lower(input_array).compile().as_text()
    hlo_original = rfftn_original.lower(input_array).compile().as_text()
    assert "dynamic-slice" not in hlo
    assert "all-gather" not in hlo
    assert "dynamic-slice" in hlo_original
    assert "all-gather" in hlo_original

    output_array = rfftn(input_array)
    sharding_info(output_array, "output_array")
    assert generalize(capsys.readouterr().out) == """
╭──────────────── input_array ─────────────────╮
│ shape: (128, 128, 128)                       │
│ dtype: float32                               │
│ size: 8.0 MiB                                │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
╭──────────────── output_array ────────────────╮
│ shape: (128, 128, 65)                        │
│ dtype: complex64                             │
│ size: 8.1 MiB                                │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
""".lstrip()

    # with shardy instead the `sharding_rule` is used to give the exact same result
    with use_shardy_partitioner(True):
        rfftn_shardy = jax.jit(
            _rfftn,
            in_shardings=simple_sharding,
            out_shardings=simple_sharding,
        )
        output_array_shardy = rfftn_shardy(input_array)
        sharding_info(output_array_shardy, "output_array_shardy")
        assert generalize(capsys.readouterr().out) == """
╭──────────── output_array_shardy ─────────────╮
│ shape: (128, 128, 65)                        │
│ dtype: complex64                             │
│ size: 8.1 MiB                                │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
""".lstrip()


def test_eval_shape(capsys):
    """
    one can also print the output of a jitted function without ever executing it by using eval_shape
    """

    def simple_function(a):
        return jax.numpy.zeros(shape=(128, 128, 128)) * a

    simple_function = jax.jit(simple_function, out_shardings=simple_sharding)

    expected_output = """
╭──────────────────────────────────────────────╮
│ shape: (128, 128, 128)                       │
│ dtype: float32                               │
│ ShapeDtypeStruct                             │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
""".lstrip()

    input_array = jax.numpy.zeros(shape=(128, 128, 128))

    output_eval: jax.ShapeDtypeStruct = simple_function.eval_shape(input_array)

    sharding_info(output_eval)
    assert generalize(capsys.readouterr().out) == expected_output

    # this even works without ever allocating the input arrays

    input_placeholder = jax.ShapeDtypeStruct((128, 128, 128), jax.numpy.float32)

    output_eval_placeholder: jax.ShapeDtypeStruct = simple_function.eval_shape(input_placeholder)

    sharding_info(output_eval_placeholder)
    assert generalize(capsys.readouterr().out) == expected_output


def test_scalar(capsys):
    """
    scalars in jax are arrays with shape (), so they should work the same
    """
    some_scalar_value = jax.numpy.asarray(42)
    sharding_info(some_scalar_value, "some_scalar_value")

    some_array = jax.numpy.array([1, 2, 3])
    sharding_info(some_array[0], "some_array[0]")

    assert generalize(capsys.readouterr().out) == """
╭─ some_scalar_value ─╮
│ shape: ()           │
│ dtype: int32        │
│ value: 42           │
│ size: 4.0 B         │
│ weakly-typed        │
│ not sharded         │
╰─────────────────────╯
╭─ some_array[0] ─╮
│ shape: ()       │
│ dtype: int32    │
│ value: 1        │
│ size: 4.0 B     │
│ not sharded     │
╰─────────────────╯
""".lstrip()


def test_numpy(capsys):
    arr = np.zeros(shape=(10, 10, 10))
    sharding_info(arr)
    with pytest.raises(ValueError, match="is not a jax array, got <class 'numpy.ndarray'>"):
        sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────╮
│ shape: (10, 10, 10) │
│ dtype: float64      │
│ size: 7.8 KiB       │
│ numpy array         │
╰─────────────────────╯
""".lstrip()


def test_2d_sharded(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, NamedSharding(mesh_2d, P(None, "a", "b")))
    sharding_info(arr)
    sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭──────────────────────────────────────────────╮
│ shape: (32, 32, 32)                          │
│ dtype: complex64                             │
│ size: 256.0 KiB                              │
│ NamedSharding: P(None, 'a', 'b')             │
│ axis 1 is sharded: CPU 0 contains 0:8 (1/4)  │
│                    Total size: 32            │
│ axis 2 is sharded: CPU 0 contains 0:16 (1/2) │
│                    Total size: 32            │
╰──────────────────────────────────────────────╯
───────────── showing dims [1, 2] from original shape (32, 32, 32) ─────────────
┌───────┬───────┐
│ CPU 0 │ CPU 1 │
├───────┼───────┤
│ CPU 2 │ CPU 3 │
├───────┼───────┤
│ CPU 4 │ CPU 5 │
├───────┼───────┤
│ CPU 6 │ CPU 7 │
└───────┴───────┘
""".lstrip()


def test_3d_sharded(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, NamedSharding(mesh_3d, P("a", "b", "c")))
    sharding_info(arr)
    with pytest.raises(NotImplementedError,
                       match=r"can only visualize up to 2 sharded dimension. \[0, 1, 2\] are sharded."):
        sharding_vis(arr)
    assert generalize(capsys.readouterr().out) == """
╭──────────────────────────────────────────────╮
│ shape: (32, 32, 32)                          │
│ dtype: complex64                             │
│ size: 256.0 KiB                              │
│ NamedSharding: P('a', 'b', 'c')              │
│ axis 0 is sharded: CPU 0 contains 0:16 (1/2) │
│                    Total size: 32            │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/2) │
│                    Total size: 32            │
│ axis 2 is sharded: CPU 0 contains 0:16 (1/2) │
│                    Total size: 32            │
╰──────────────────────────────────────────────╯
""".lstrip()


def test_shard_map(capsys):
    """
    https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
    """
    arr = jax.numpy.zeros(shape=(16, 16))

    @partial(shard_map, mesh=mesh, in_specs=P(None, 'gpus'), out_specs=P(None, 'gpus'))
    def test(a):
        # sharding_info(a,"input") # doesn't seem to work inside a shard_map
        return a ** 2

    out = test(arr)

    sharding_info(out)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16)                             │
│ dtype: float32                              │
│ size: 1.0 KiB                               │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_simple_array_info(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, simple_sharding)
    simple_array_info(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:4 (1/8) │
│                    Total size: 32           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_inside_shard_map_failing(capsys):
    arr = jax.numpy.zeros(shape=(16, 16))

    def test(a):
        sharding_info(a, "input")
        return a ** 2

    with pytest.raises(NotImplementedError) as e_info:
        func_shard_map = shard_map(test, mesh=mesh, in_specs=P(None, 'gpus'), out_specs=P(None, 'gpus'))
        out = func_shard_map(arr)


def test_inside_shard_map_simple(capsys):
    arr = jax.numpy.zeros(shape=(16, 16))

    def test(a):
        simple_array_info(a, "input")
        return a ** 2

    func_shard_map = shard_map(test, mesh=mesh, in_specs=P(None, 'gpus'), out_specs=P(None, 'gpus'))
    out = func_shard_map(arr)
    assert generalize(capsys.readouterr().out) == """
╭──── input ─────╮
│ shape: (16, 2) │
│ dtype: float32 │
│ size: 128.0 B  │
│ called in jit  │
╰────────────────╯
""".lstrip()


def test_indirectly_sharded(capsys):
    """
    y is never explicitly sharded, but it seems like the sharding is back-propagated through the jit compiled function
    """
    arr = jax.numpy.zeros(shape=(16, 16, 16))

    def func(x):
        y = jax.numpy.zeros(shape=(16, 16, 16))
        sharding_info(y)
        return x * y

    func = jax.jit(func, out_shardings=simple_sharding)
    arr = func(arr)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16, 16)                         │
│ dtype: float32                              │
│ size: 16.0 KiB                              │
│ called in jit                               │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()

    # doing this with shardy seems to not print the back-propagated sharding
    with use_shardy_partitioner(True):
        func = jax.jit(func, out_shardings=simple_sharding)
        arr = func(arr)
        sharding_info(arr, "output")
        assert generalize(capsys.readouterr().out) == """
╭─────────────────────╮
│ shape: (16, 16, 16) │
│ dtype: float32      │
│ size: 16.0 KiB      │
│ called in jit       │
│ NamedSharding: P()  │
╰─────────────────────╯
╭────────────────── output ───────────────────╮
│ shape: (16, 16, 16)                         │
│ dtype: float32                              │
│ size: 16.0 KiB                              │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_with_sharding_constraint(capsys):
    arr = jax.numpy.zeros(shape=(16, 16, 16))

    def func(x):
        return jax.lax.with_sharding_constraint(x, simple_sharding)

    func = jax.jit(func)
    arr = func(arr)
    sharding_info(arr)

    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16, 16)                         │
│ dtype: float32                              │
│ size: 16.0 KiB                              │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_sharding_outer_product(capsys):
    arr = jax.numpy.arange(16)
    arr = jax.device_put(arr, NamedSharding(mesh, P("gpus")))

    product = jax.numpy.outer(arr, arr)
    assert product.shape == (16, 16)

    sharding_info(product)

    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16)                             │
│ dtype: int32                                │
│ size: 1.0 KiB                               │
│ NamedSharding: P('gpus',)                   │
│ axis 0 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_sharded_closure(capsys):
    """
    While not great code style, sharding also works in constant arrays
    that are not passed to functions.

    This does not work in a multihost setup.
    """
    arr = jax.numpy.zeros((16, 16))
    arr = jax.device_put(arr, NamedSharding(mesh, P("gpus")))

    def some_function():
        return arr * 5

    some_function = jax.jit(some_function, out_shardings=simple_sharding)

    out = some_function()
    sharding_info(out)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16)                             │
│ dtype: float32                              │
│ size: 1.0 KiB                               │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_nondefault_device(capsys):
    some_array = jax.numpy.zeros(16)
    some_array_on_one_device = jax.device_put(some_array, devices[2])
    sharding_info(some_array_on_one_device)

    assert generalize(capsys.readouterr().out) == """
╭────────────────────╮
│ shape: (16,)       │
│ dtype: float32     │
│ size: 64.0 B       │
│ device: TFRT_CPU_2 │
│ not sharded        │
╰────────────────────╯
""".lstrip()


def test_device_put_replicated(capsys):
    some_array = jax.numpy.zeros(16)
    replicated_array = jax.device_put_replicated(some_array, jax.devices())
    sharding_info(replicated_array)

    assert generalize(capsys.readouterr().out) == """
╭──────────────────────────────────────────────────────────────────────╮
│ shape: (8, 16)                                                       │
│ dtype: float32                                                       │
│ size: 512.0 B                                                        │
│ PmapSharding(sharding_spec=ShardingSpec((Chunked(8), NoSharding()),  │
│ (ShardedAxis(axis=0),)), device_ids=[0, 1, 2, 3, 4, 5, 6, 7],        │
│ device_platform=CPU, device_shape=(8,))                              │
│ axis 0 is sharded: CPU 0 contains 0:1 (1/8)                          │
│                    Total size: 8                                     │
╰──────────────────────────────────────────────────────────────────────╯
""".lstrip()


def test_device_put_sharded(capsys):
    list_of_arrays = []
    for i in range(len(jax.devices())):
        some_array = jax.numpy.full((3,), i)
        list_of_arrays.append(some_array)
    replicated_array = jax.device_put_sharded(list_of_arrays, jax.devices())
    sharding_info(replicated_array)
    reference = jax.numpy.array([[0, 0, 0],
                                 [1, 1, 1],
                                 [2, 2, 2],
                                 [3, 3, 3],
                                 [4, 4, 4],
                                 [5, 5, 5],
                                 [6, 6, 6],
                                 [7, 7, 7]])
    assert jax.numpy.all(replicated_array == reference)

    assert generalize(capsys.readouterr().out) == """
╭──────────────────────────────────────────────────────────────────────╮
│ shape: (8, 3)                                                        │
│ dtype: int32                                                         │
│ size: 96.0 B                                                         │
│ weakly-typed                                                         │
│ PmapSharding(sharding_spec=ShardingSpec((Chunked(8), NoSharding()),  │
│ (ShardedAxis(axis=0),)), device_ids=[0, 1, 2, 3, 4, 5, 6, 7],        │
│ device_platform=CPU, device_shape=(8,))                              │
│ axis 0 is sharded: CPU 0 contains 0:1 (1/8)                          │
│                    Total size: 8                                     │
╰──────────────────────────────────────────────────────────────────────╯
""".lstrip()


def test_make_array_from_single_device_arrays(capsys):
    list_of_arrays = []
    for i, device in enumerate(jax.devices()):
        some_array = jax.numpy.full((3, 2), i)
        some_array = jax.device_put(some_array, device)
        list_of_arrays.append(some_array)
    replicated_array = jax.make_array_from_single_device_arrays((3, 16), simple_sharding, list_of_arrays)
    sharding_info(replicated_array)
    assert generalize(capsys.readouterr().out) == """
╭─────────────────────────────────────────────╮
│ shape: (3, 16)                              │
│ dtype: int32                                │
│ size: 192.0 B                               │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_array_stats(capsys):
    for buf in jax.live_arrays(): buf.delete()
    arr = jax.numpy.zeros(shape=(16, 16, 16))
    arr2 = jax.device_put(jax.numpy.zeros(shape=(2, 16, 4)), simple_sharding)

    some_scalar_value1 = jax.numpy.array(4)
    some_scalar_value2 = jax.numpy.array(42)
    some_scalar_value3 = jax.numpy.array(42.4)

    print_array_stats()

    assert generalize(capsys.readouterr().out) == """
                  allocated jax arrays                   
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ size     ┃ shape        ┃ dtype   ┃      sharded      ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ 16.0 KiB │ (16, 16, 16) │ float32 │                   │
│ 64.0 B   │ (2, 16, 4)   │ float32 │ ✔ (512.0 B total) │
├──────────┼──────────────┼─────────┼───────────────────┤
│ 4.0 B    │ 1×s          │ float32 │                   │
│ 8.0 B    │ 2×s          │ int32   │                   │
├──────────┼──────────────┼─────────┼───────────────────┤
│ 16.1 KiB │              │         │                   │
└──────────┴──────────────┴─────────┴───────────────────┘
""".lstrip("\n")


def test_array_stats_with_labels(capsys):
    for buf in jax.live_arrays(): buf.delete()
    some_array = jax.device_put(jax.numpy.zeros(shape=(16, 16, 16)), simple_sharding)

    sharding_info(some_array, "some_array")
    print_array_stats()
    assert generalize(capsys.readouterr().out) == """
╭──────────────── some_array ─────────────────╮
│ shape: (16, 16, 16)                         │
│ dtype: float32                              │
│ size: 16.0 KiB                              │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
                         allocated jax arrays                         
┏━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ size    ┃ shape        ┃ dtype   ┃      sharded       ┃ label      ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 2.0 KiB │ (16, 16, 16) │ float32 │ ✔ (16.0 KiB total) │ some_array │
├─────────┼──────────────┼─────────┼────────────────────┼────────────┤
│ 2.0 KiB │              │         │                    │            │
└─────────┴──────────────┴─────────┴────────────────────┴────────────┘
""".lstrip("\n")


def test_array_stats_with_small_arrays_hidden(capsys):
    for buf in jax.live_arrays(): buf.delete()
    some_array = jax.device_put(jax.numpy.zeros(shape=(128, 16, 16)), simple_sharding)
    small_array = jax.numpy.zeros(shape=(100))
    small_array2 = jax.numpy.zeros(shape=(500))

    print_array_stats(hide_small_arrays=True)
    assert generalize(capsys.readouterr().out) == """
                    allocated jax arrays                    
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ size     ┃ shape         ┃ dtype   ┃       sharded       ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ 16.0 KiB │ (128, 16, 16) │ float32 │ ✔ (128.0 KiB total) │
│ 2.3 KiB  │ small arrays  │         │                     │
├──────────┼───────────────┼─────────┼─────────────────────┤
│ 18.3 KiB │               │         │                     │
└──────────┴───────────────┴─────────┴─────────────────────┘
""".lstrip("\n")


def test_mesh_variable():
    """
    Different Mesh objects are equivalent as long as they have the same properties.
    Therefore, one should be able to create new one instead of using the global one
    without changing anything.
    """

    def get_mesh() -> Mesh:
        num_gpus = jax.device_count()
        devices = mesh_utils.create_device_mesh((num_gpus,))
        return Mesh(devices, axis_names=('gpus',))

    own_mesh_obj = get_mesh()
    assert own_mesh_obj == mesh


def test_non_array(capsys):
    arr = [1, 2, 3]
    with pytest.raises(ValueError, match="is not a jax array, got <class 'list'>"):
        sharding_info(arr)
    with pytest.raises(ValueError, match="is not a jax array, got <class 'list'>"):
        sharding_vis(arr)

    # allow printing some primitive types
    sharding_info("test","some string")
    sharding_info(123,"some integer")
    sharding_info(float(np.pi),"some float")
    sharding_info(True,"some boolean")
    sharding_info(np.array([1])[0],"some numpy scalar")
    sharding_info(None,"None")
    assert generalize(capsys.readouterr().out) == """
╭─ some string ─╮
│ type: str     │
│ value: test   │
╰───────────────╯
╭─ some integer ─╮
│ type: int      │
│ value: 123     │
╰────────────────╯
╭─────── some float ───────╮
│ type: float              │
│ value: 3.141592653589793 │
╰──────────────────────────╯
╭─ some boolean ─╮
│ type: bool     │
│ value: True    │
╰────────────────╯
╭─ some numpy scalar ─╮
│ type: int64         │
│ value: 1            │
╰─────────────────────╯
╭─ None ─╮
│ None   │
╰────────╯
""".lstrip("\n")
