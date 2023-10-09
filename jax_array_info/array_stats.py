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
    table.add_column("sharded", justify="center")
    total_size = 0
    for arr in array_stats_data():
        file_size = arr.nbytes
        is_sharded = False
        if len(arr.sharding.device_set)>1:
            file_size /= len(arr.sharding.device_set)
            is_sharded = True
        total_size += file_size
        table.add_row(pretty_byte_size(file_size), str(arr.shape), f"✔ ({pretty_byte_size(arr.nbytes)} total)" if is_sharded else "")
    table.add_section()
    table.add_row(pretty_byte_size(total_size))
    console.print(table)
