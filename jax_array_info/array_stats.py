from collections import Counter

import jax
import rich
from rich.table import Table

from .utils import pretty_byte_size, config


def array_stats_data() -> list[jax.Array]:
    arrs = jax.live_arrays()
    arrs.sort(key=lambda a: -a.nbytes)
    return arrs


def print_array_stats(hide_small_arrays: bool = False):
    multiple_devices = len(jax.devices()) > 1
    console = rich.console.Console()
    table = Table(title="allocated jax arrays")
    table.add_column("size")
    table.add_column("shape")
    table.add_column("dtype")
    if multiple_devices:
        table.add_column("sharded", justify="center")
    any_label_exists = False
    if config.assign_labels_to_arrays:
        for arr in array_stats_data():
            if hasattr(arr, "_custom_label"):
                any_label_exists = True
        if any_label_exists:
            table.add_column("label")
    scalar_stats = Counter()
    scalar_size_stats = Counter()
    total_size = 0
    skipped_size = 0
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
        columns = [
            pretty_byte_size(file_size),
            str(arr.shape),
            str(arr.dtype),
        ]
        if multiple_devices:
            columns.append(f"✔ ({pretty_byte_size(arr.nbytes)} total)" if is_sharded else "")

            label = ""
        if config.assign_labels_to_arrays and any_label_exists:
            if hasattr(arr, "_custom_label"):
                label = arr._custom_label
            columns.append(label)
        if hide_small_arrays and (not label) and file_size < (1024 * 3):
            skipped_size += file_size
            continue
        table.add_row(*columns),
    if skipped_size > 0:
        table.add_row(
            pretty_byte_size(skipped_size),
            "small arrays",
            ""
        )
    if len(scalar_stats):
        table.add_section()
        for dtype, count in scalar_stats.items():
            table.add_row(
                pretty_byte_size(scalar_size_stats[dtype]),
                f"{count}×s",
                str(dtype),
            )
    table.add_section()
    table.add_row(pretty_byte_size(total_size))
    console.print(table)
