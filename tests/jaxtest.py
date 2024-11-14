import os
from functools import partial

import jax.numpy
import numpy as np
import pytest
from jax._src.sharding_impls import PositionalSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_array_info import sharding_info, sharding_vis, print_array_stats, simple_array_info

num_gpus = 8

os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_gpus}'

devices = mesh_utils.create_device_mesh((num_gpus,))
mesh = Mesh(devices, axis_names=('gpus',))

simple_sharding = NamedSharding(mesh, P(None, "gpus"))

devices_2d = mesh_utils.create_device_mesh((num_gpus // 2, 2))
mesh_2d = Mesh(devices_2d, axis_names=('a', 'b'))
devices_3d = mesh_utils.create_device_mesh((num_gpus // 4, 2, 2))
mesh_3d = Mesh(devices_3d, axis_names=('a', 'b', 'c'))


def test_simple(capsys):
    arr = jax.numpy.array([1, 2, 3])
    sharding_info(arr, "arr")
    sharding_vis(arr)

    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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


def test_device_put_sharded(capsys):
    arr = jax.numpy.zeros(shape=(8 * 4, 8 * 4, 8 * 4), dtype=jax.numpy.complex64)
    arr = jax.device_put(arr, simple_sharding)
    sharding_info(arr)
    sharding_vis(arr)
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
╭─────────────────────────────────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ called in jit                               │
│ PositionalSharding:                         │
│ [[[{CPU 0}]                                 │
│   [{CPU 1}]                                 │
│   [{CPU 2}]                                 │
│   [{CPU 3}]                                 │
│   [{CPU 4}]                                 │
│   [{CPU 5}]                                 │
│   [{CPU 6}]                                 │
│   [{CPU 7}]]]                               │
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

    assert capsys.readouterr().out == """
╭──────────────────────────────────────────────────────────────────────╮
│ shape: (8, 24)                                                       │
│ dtype: complex64                                                     │
│ size: 1.5 KiB                                                        │
│ PmapSharding(sharding_spec=ShardingSpec((Chunked(8), NoSharding()),  │
│ (ShardedAxis(axis=0),)), device_ids=[0, 1, 2, 3, 4, 5, 6, 7],        │
│ device_platform=CPU, device_shape=(8,))                              │
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


def test_numpy(capsys):
    arr = np.zeros(shape=(10, 10, 10))
    sharding_info(arr)
    with pytest.raises(ValueError, match="is not a jax array, got <class 'numpy.ndarray'>"):
        sharding_vis(arr)
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
╭─────────────────────╮
│ shape: (32, 32, 32) │
│ dtype: complex64    │
│ size: 256.0 KiB     │
╰─────────────────────╯
""".lstrip()


def test_inside_shard_map(capsys):
    arr = jax.numpy.zeros(shape=(16, 16))

    def test(a):
        sharding_info(a, "input")
        return a ** 2

    with pytest.raises(NotImplementedError) as e_info:
        func_shard_map = shard_map(test, mesh=mesh, in_specs=P(None, 'gpus'), out_specs=P(None, 'gpus'))
        out = func_shard_map(arr)


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
    assert capsys.readouterr().out == """
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
    assert capsys.readouterr().out == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16, 16)                         │
│ dtype: float32                              │
│ size: 16.0 KiB                              │
│ called in jit                               │
│ PositionalSharding:                         │
│ [[[{CPU 0}]                                 │
│   [{CPU 1}]                                 │
│   [{CPU 2}]                                 │
│   [{CPU 3}]                                 │
│   [{CPU 4}]                                 │
│   [{CPU 5}]                                 │
│   [{CPU 6}]                                 │
│   [{CPU 7}]]]                               │
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

    assert capsys.readouterr().out == """
╭─────────────────────────────────────────────╮
│ shape: (16, 16, 16)                         │
│ dtype: float32                              │
│ size: 16.0 KiB                              │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:2 (1/8) │
│                    Total size: 16           │
╰─────────────────────────────────────────────╯
""".lstrip()


def test_array_stats(capsys):
    for buf in jax.live_arrays(): buf.delete()
    arr = jax.numpy.zeros(shape=(16, 16, 16))
    arr2 = jax.device_put(jax.numpy.zeros(shape=(2, 16, 4)), simple_sharding)

    print_array_stats()

    assert capsys.readouterr().out == """
                  allocated jax arrays                   
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ size     ┃ shape        ┃ dtype   ┃      sharded      ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ 16.0 KiB │ (16, 16, 16) │ float32 │                   │
│ 64.0 B   │ (2, 16, 4)   │ float32 │ ✔ (512.0 B total) │
├──────────┼──────────────┼─────────┼───────────────────┤
│ 16.1 KiB │              │         │                   │
└──────────┴──────────────┴─────────┴───────────────────┘
""".lstrip("\n")


def test_non_array(capsys):
    arr = [1, 2, 3]
    with pytest.raises(ValueError, match="is not a jax array, got <class 'list'>"):
        sharding_info(arr)
    with pytest.raises(ValueError, match="is not a jax array, got <class 'list'>"):
        sharding_vis(arr)


if __name__ == '__main__':
    test_indirectly_sharded(None)
