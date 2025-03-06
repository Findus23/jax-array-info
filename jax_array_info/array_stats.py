from collections import Counter

import jax
import rich
from jax.sharding import SingleDeviceSharding
from rich.table import Table

from .utils import pretty_byte_size


def array_stats_data() -> list[jax.Array]:
    arrs = jax.live_arrays()
    arrs.sort(key=lambda a: -a.nbytes)
    return arrs


def print_array_stats():
    console = rich.console.Console()
    table = Table(title="allocated jax arrays")
    table.add_column("size")
    table.add_column("shape")
    table.add_column("dtype")
    table.add_column("sharded", justify="center")
    scalar_stats = Counter()
    scalar_size_stats = Counter()
    total_size = 0
    for arr in array_stats_data():
        file_size = arr.nbytes
        is_sharded = False
        if len(arr.sharding.device_set) > 1:
            file_size /= len(arr.sharding.device_set)
            is_sharded = True
        total_size += file_size
        if len(arr.shape) == 0:
            scalar_stats[arr.dtype] += 1
            scalar_size_stats[arr.dtype] += file_size
            continue
        table.add_row(
            pretty_byte_size(file_size),
            str(arr.shape),
            str(arr.dtype),
            f"✔ ({pretty_byte_size(arr.nbytes)} total)" if is_sharded else "")
    if len(scalar_stats):
        table.add_section()
        for dtype,count in scalar_stats.items():
            table.add_row(
                pretty_byte_size(scalar_size_stats[dtype]),
                f"{count}×s",
                str(dtype),
            )
    table.add_section()
    table.add_row(pretty_byte_size(total_size))
    console.print(table)
