import os

import rich
from jax import Device
from rich.table import Table

from .utils import pretty_byte_size


def pretty_memory_stats(device: Device, override=None):
    console = rich.console.Console()
    if override is None:
        stats = device.memory_stats()
    else:
        stats = override
    if stats is None:
        if device.device_kind == "cpu":
            reason = "cpu device"
        if os.getenv("XLA_PYTHON_CLIENT_ALLOCATOR", None) == "platform":
            reason = "ALLOCATOR=platform"
        console.print(f"[red]No memory stats for {device} ({reason}?)")
        return
    num_allocs = stats.pop('num_allocs')
    table = Table(title=f"Memory Stats of {device}\n{num_allocs} allocs")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("size (raw)", justify="right")

    for k, v in override.items():
        table.add_row(
            k,
            pretty_byte_size(v),
            str(v),
        )
    console.print(table)
