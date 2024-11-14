from typing import Optional

import numpy as np
import rich
from jax import Array
from jax._src.debugging import inspect_array_sharding
from jax.core import Tracer
from jax.sharding import Sharding, NamedSharding, GSPMDSharding, SingleDeviceSharding, PmapSharding, PositionalSharding
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .utils import pretty_byte_size

SupportedArray = np.ndarray | Array


def sharding_info(arr: SupportedArray, name: str = None):
    if isinstance(arr, np.ndarray):
        return print_sharding_info(arr, None, name)
    if not isinstance(arr, Array):
        raise ValueError(f"is not a jax array, got {type(arr)}")

    def _info(sharding):
        print_sharding_info(arr, sharding, name)

    inspect_array_sharding(arr, callback=_info)

def simple_array_info(arr: SupportedArray, name: str = None):
    return print_sharding_info(arr, None, name)


def _print_sharding_info_raw(arr: SupportedArray, sharding: Optional[Sharding], console: Console):
    shape = arr.shape
    console.print(f"shape: {shape}")
    console.print(f"dtype: {arr.dtype}")
    console.print(f"size: {pretty_byte_size(arr.nbytes)}")

    if isinstance(arr, np.ndarray):
        console.print("[bold]numpy array")
        return
    if not isinstance(arr, Array):
        raise ValueError(f"is not a jax array, got {type(arr)}")

    is_in_jit = isinstance(arr, Tracer)
    if not is_in_jit and not arr.is_fully_addressable:
        console.print("!is_fully_addressable")
    if is_in_jit:
        console.print("[bright_black]called in jit")
    # if not arr.is_fully_replicated:
    #     console.print("!is_fully_replicated")
    if sharding is None:
        return
    device_kind = next(iter(sharding.device_set)).platform.upper()
    if isinstance(sharding, SingleDeviceSharding):
        console.print("[red]not sharded")
    if isinstance(sharding, GSPMDSharding):
        console.print(sharding)
    if isinstance(sharding, PositionalSharding):
        console.print(f"PositionalSharding:")
        console.print(sharding._ids)
    if isinstance(sharding, NamedSharding):
        console.print(f"NamedSharding: P{tuple(sharding.spec)}")

    if isinstance(sharding, PmapSharding):
        console.print(sharding)
        return
    device_indices_map = sharding.devices_indices_map(tuple(shape))
    slcs = next(iter(device_indices_map.values()))
    sl: slice
    for i, sl in enumerate(slcs):
        if sl.start is None:
            continue
        local_size = sl.stop - sl.start
        global_size = shape[i]
        num_shards = global_size // local_size
        console.print(f"axis {i} is sharded: {device_kind} 0 contains {sl.start}:{sl.stop} (1/{num_shards})")
        console.print(f"                   Total size: {global_size}")


def print_sharding_info(arr: SupportedArray, sharding: Optional[Sharding], name: str = None):
    console = rich.console.Console()
    with console.capture() as capture:
        _print_sharding_info_raw(arr, sharding, console)
    str_output = capture.get()
    text = Text.from_ansi(str_output)
    console.print(Panel(text, expand=False, title=f"[bold]{name}" if name is not None else None))
