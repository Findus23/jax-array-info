# jax-array-info

This package contains two functions for debugging jax `Array`s:

```python
from jax_array_info import sharding_info, sharding_vis
```

## `sharding_info(arr)`

`sharding_info(arr)` prints general information about a jax or numpy array with special focus on sharding (
supporting `SingleDeviceSharding`, `GSPMDSharding`, `PositionalSharding`, `NamedSharding` and `PmapSharding`)

```python
array = jax.numpy.zeros(shape=(N, N, N), dtype=jax.numpy.float32)
array = jax.device_put(array, NamedSharding(mesh, P(None, "gpus")))
sharding_info(array, "some_array")
```

```text
╭────────────────── some_array ───────────────────╮
│ shape: (128, 128, 128)                          │
│ dtype: float32                                  │
│ size: 8.0 MiB                                   │
│ NamedSharding: P(None, 'gpus')                  │
│ axis 1 is sharded: CPU 0 contains 0:16 (of 128) │
╰─────────────────────────────────────────────────╯
```

## `sharding_vis(arr)`

A modified version
of [`jax.debug.visualize_array_sharding()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.visualize_array_sharding.html)
that also supports arrays with more than 2 dimensions (by ignoring non-sharded dimensions in the visualisation until
reaching 2 dimensions)

```python
array = jax.numpy.zeros(shape=(N, N, N), dtype=jax.numpy.float32)
array = jax.device_put(array, NamedSharding(mesh, P(None, "gpus")))
sharding_vis(array)
```

```text
─────────── showing dims [0, 1] from original shape (128, 128, 128) ────────────
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
```

### Examples

See [`tests/`](./tests/jaxtest.py)
