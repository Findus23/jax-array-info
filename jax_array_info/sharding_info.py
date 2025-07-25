from typing import Optional

import jax
import numpy as np
import rich
from jax import Array, Device
from jax.core import Tracer
from jax.debug import inspect_array_sharding
from jax.sharding import Sharding, NamedSharding, SingleDeviceSharding, PmapSharding
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .utils import pretty_byte_size, config

SupportedArray = np.ndarray | Array


def print_trivial(title: str, content: str):
    console = rich.console.Console()
    console.print(Panel(
        content,
        expand=False,
        title=f"[bold]{title}" if title is not None else None,
    ))


def sharding_info(arr: SupportedArray, name: str = None):
    if config.assign_labels_to_arrays and isinstance(arr, Array) and not isinstance(arr, Tracer):
        arr._custom_label = name
    if isinstance(arr, np.ndarray):
        return print_sharding_info(arr, None, name)
    if not isinstance(arr, (Array, jax.ShapeDtypeStruct)):
        if arr is None:
            print_trivial(name, "None")
            return
        if isinstance(arr, list):
            print_trivial(name, f"type: list\nlen: {len(arr)}")
            return
        if np.ndim(arr) == 0:
            print_trivial(name, f"type: {type(arr).__name__}\nvalue: {arr}")
            return
        raise ValueError(f"is not a jax array, got {type(arr)}")

    def _info(sharding):
        print_sharding_info(arr, sharding, name)

    inspect_array_sharding(arr, callback=_info)


def label_array(arr: Array, name: str):
    if config.assign_labels_to_arrays and isinstance(arr, Array) and not isinstance(arr, Tracer):
        arr._custom_label = name


def simple_array_info(arr: SupportedArray, name: str = None):
    sharding = None
    if isinstance(arr, Array):
        try:
            # if we are outside of jit, we can also read the sharding directly without inspect_array_sharding()
            # and therefore still show it even with simple_array_info()
            sharding = arr.sharding
        except AttributeError:
            sharding = None
    return print_sharding_info(arr, sharding, name)


def _print_sharding_info_raw(arr: SupportedArray, sharding: Optional[Sharding], console: Console):
    shape = arr.shape
    sharded_axes = set()
    vma_axes = set()
    # console.print(f"shape: {shape}")
    console.print(f"dtype: {arr.dtype}")
    if len(shape) == 0:
        console.print(f"value: {arr}")
    if hasattr(arr, "nbytes"):
        # missing in ShapeDtypeStruct
        console.print(f"size: {pretty_byte_size(arr.nbytes)}")

    if isinstance(arr, np.ndarray):
        console.print("[bold]numpy array")
        return sharded_axes, vma_axes
    if not isinstance(arr, (Array, jax.ShapeDtypeStruct)):
        raise ValueError(f"is not a jax array, got {type(arr)}")

    default_device = jax.devices()[0]
    is_a_shapedtypestruct = isinstance(arr, jax.ShapeDtypeStruct)
    is_in_jit = isinstance(arr, Tracer)
    if not is_in_jit and not is_a_shapedtypestruct and not arr.is_fully_addressable:
        console.print("!is_fully_addressable")
    if is_in_jit:
        console.print("[bright_black]called in jit")
    if is_a_shapedtypestruct:
        console.print("[bright_black]ShapeDtypeStruct")
    if arr.weak_type:
        console.print("weakly-typed")
    if hasattr(arr, "device") and isinstance(arr.device, Device) and arr.device != default_device:
        console.print(f"[bright_black]device: {arr.device}")
    # if not arr.is_fully_replicated:
    #     console.print("!is_fully_replicated")
    if sharding is None:
        return sharded_axes, vma_axes

    if hasattr(arr, "vma") and arr.vma:
        console.print(f"vma: {set(arr.vma)}")

    device_kind = next(iter(sharding.device_set)).platform.upper()

    if device_kind == "gpu" and sharding.memory_kind != "device":
        # print non-standard memory_kind
        console.print(f"memory_kind: {sharding.memory_kind}")

    if isinstance(sharding, SingleDeviceSharding):
        if len(jax.devices()) > 1:
            # only warn about missing sharding if multiple devices are used
            console.print("[red]not sharded")
    if isinstance(sharding, NamedSharding):
        console.print(f"NamedSharding: P{tuple(sharding.spec)}")
        if hasattr(arr, "vma") and arr.vma:
            for i, val in enumerate(sharding.spec):
                if val in arr.vma:
                    vma_axes.add(i)

    if isinstance(sharding, PmapSharding):
        console.print(sharding)
        # return
    device_indices_map = sharding.devices_indices_map(tuple(shape))
    slcs = next(iter(device_indices_map.values()))
    sl: slice
    for i, sl in enumerate(slcs):
        if sl.start is None:
            continue
        sharded_axes.add(i)
        local_size = sl.stop - sl.start
        global_size = shape[i]
        num_shards = global_size // local_size
        console.print(f"axis {i} is sharded: {device_kind} 0 contains {sl.start}:{sl.stop} (1/{num_shards})")
        console.print(f"                   Total size: {global_size}")

    return sharded_axes, vma_axes



def format_shape(shape: tuple[int, ...], sharded_axes: set[int], vma_axes:set[int]):
    text = [("(", "blue")]
    for i,dim in enumerate(shape):
        style = "bold magenta"
        if i in sharded_axes:
            style = "bold green"
        if i in vma_axes:
            style = "bold bright_green"
        text.append((str(dim), style))
        if i !=len(shape) - 1:
            text.append((", ", ""))
        elif len(shape) == 1:
            text.append((",", ""))
    text.append((")", "blue"))
    return Text.assemble(*text)

def print_sharding_info(arr: SupportedArray, sharding: Optional[Sharding], name: str = None):
    console = rich.console.Console()
    with console.capture() as capture:
        sharded_axes, vma_axes = _print_sharding_info_raw(arr, sharding, console)
    str_output = capture.get()
    text = Text.from_ansi(str_output)
    with console.capture() as capture:
        shape_text = format_shape(arr.shape, sharded_axes, vma_axes)
        console.print("shape:",shape_text)
    str_output = capture.get()
    text_shape = Text.from_ansi(str_output)

    console.print(Panel(text_shape + "\n" + text, expand=False, title=f"[bold]{name}" if name is not None else None))
