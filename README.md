# jax-array-info

This package contains two functions for debugging jax `Array`s:

```bash
pip install git+https://github.com/Findus23/jax-array-info.git
```

```python
from jax_array_info import sharding_info, sharding_vis, simple_array_info, print_array_stats, pretty_memory_stats
```

## `sharding_info(arr)`

`sharding_info(arr)` prints general information about a jax or numpy array with special focus on sharding (
supporting `SingleDeviceSharding`, `GSPMDSharding`, `PositionalSharding`, `NamedSharding` and `PmapSharding`)

```python
some_array = jax.numpy.zeros(shape=(N, N, N), dtype=jax.numpy.float32)
some_array = jax.device_put(some_array, NamedSharding(mesh, P(None, "gpus")))
sharding_info(some_array, "some_array")
```

```text
╭───────────────── some_array ─────────────────╮
│ shape: (128, 128, 128)                       │
│ dtype: float32                               │
│ size: 8.0 MiB                                │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
```

## `simple_array_info()`

`sharding_info()` uses a jax callback to make sure it can get sharding information in as many situations as possible.
But this means it is broken when used in some type of functions (e.g. when [using
`shard_map`](https://github.com/jax-ml/jax/issues/23936)). `simple_array_info()` gives the same output, with the
advantage of working everywhere (it is equivalent to a `print`) and the tradeoff that it is not guaranteed to always
report correct sharding information (e.g. inside jitted functions).

## `print_array_stats()`

Shows a nice overview over the all currently allocated arrays ordered by their size. To save space, scalar values are
grouped by dtype.

**Disclaimer**: This uses `jax.live_arrays()` to get its information. There might be allocated arrays that are missing
in this view. Also keep in mind that this only reports arrays that are acutally allocated, so calling it inside a jitted
function won't give useful information.

```python
arr = jax.numpy.zeros(shape=(16, 16, 16))
arr2 = jax.device_put(jax.numpy.zeros(shape=(2, 16, 4)), NamedSharding(mesh, P(None, "gpus")))
scalar = jax.numpy.array(42)

print_array_stats()
```

```text
             allocated jax arrays              
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ size     ┃ shape        ┃ dtype   ┃      sharded      ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ 16.0 KiB │ (16, 16, 16) │ float32 │                   │
│ 64.0 B   │ (2, 16, 4)   │ float32 │ ✔ (512.0 B total) │
├──────────┼──────────────┼─────────┼───────────────────┤
│ 4.0 B    │ 1×s          │ int32   │                   │
├──────────┼──────────────┼─────────┼───────────────────┤
│ 16.1 KiB │              │         │                   │
└──────────┴──────────────┴─────────┴───────────────────┘
```

## `sharding_vis(arr)`

A modified version
of [
`jax.debug.visualize_array_sharding()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.visualize_array_sharding.html)
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

## `pretty_memory_stats(device)`

This is printing the output of `device.memory_stats()` as a table. The data (and if any data is available) depends on
the device

```python
pretty_memory_stats(jax.local_devices()[0])
```

```text
               Memory Stats of cuda:0                
                     222 allocs                      
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ name                     ┃     size ┃  size (raw) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ bytes_in_use             │  3.6 GiB │  3897557504 │
│ peak_bytes_in_use        │ 29.9 GiB │ 32087482368 │
│ largest_alloc_size       │ 23.5 GiB │ 25232940032 │
│ bytes_limit              │ 33.3 GiB │ 35714334720 │
│ bytes_reserved           │    0.0 B │           0 │
│ peak_bytes_reserved      │    0.0 B │           0 │
│ largest_free_block_bytes │    0.0 B │           0 │
│ pool_bytes               │ 33.3 GiB │ 35714334720 │
│ peak_pool_bytes          │ 33.3 GiB │ 35714334720 │
└──────────────────────────┴──────────┴─────────────┘
```

## Examples

You can find many examples of how arrays can be sharded in jax and how the output would look like in [
`tests/test_jax.py`](./tests/test_jax.py). For examples of sharding arrays along multiple jax processes check [
`test_multihost.py`](./tests/test_multihost.py) and [`multihost.py`](./tests/multihost.py)

## Assigning Labels to Jax Arrays

While the output of `print_array_stats()` is very useful to see which arrays are currently allocated, it is missing the
connection to where the array is currently used. While in many cases the shape allows to guess which array a row is
refering to, it would be nice to also know its name.
Unfortunately we only get a reference to all currently allocated arrays from `jax.live_arrays()` which means we don't
know where the array is assigned and what the variable is called.
We can do an ugly workaround to still get more useful names in the `print_array_stats()` output:
Whenever `simple_array_info` is used like `simple_array_info(some_array,"some_array")`, we will modify the `some_array`
jax array by setting `some_array._custom_label = "some_array"`. Therefore this label becomes a part of the jax `Array`
object and we can use this label in our array overview.

This of course has some limitations (it only works if `simple_array_info` is used outside of jitted code) and is an ugly
hack, so it possibly will break some things. Therefore this feature can be disabled globally using

```python
from jax_array_info import sharding_info, print_array_stats, config as array_info_config

array_info_config.assign_labels_to_arrays = False
```

#### Example

```python
import jax
from jax_array_info import sharding_info, print_array_stats

some_array = jax.numpy.zeros(shape=(128, 128, 128))
sharding_info(some_array, "some_array")

print_array_stats()
```

```text
╭────── some_array ──────╮
│ shape: (128, 128, 128) │
│ dtype: float32         │
│ size: 8.0 MiB          │
│ not sharded            │
╰────────────────────────╯
                     allocated jax arrays                     
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ size    ┃ shape           ┃ dtype   ┃ sharded ┃ label      ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ 8.0 MiB │ (128, 128, 128) │ float32 │         │ some_array │
├─────────┼─────────────────┼─────────┼─────────┼────────────┤
│ 8.0 MiB │                 │         │         │            │
└─────────┴─────────────────┴─────────┴─────────┴────────────┘
```
