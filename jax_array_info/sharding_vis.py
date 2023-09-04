"""
based on visualize_sharding() from jax/_src/debugging.py

https://github.com/google/jax/blob/main/jax/_src/debugging.py

# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
from typing import Sequence, Optional, Dict, Tuple, Set

import rich
from jax import Array
from jax._src.debugging import _raise_to_slice, _slice_to_chunk_idx, inspect_array_sharding, ColorMap, make_color_iter, \
    _canonicalize_color, _get_text_color
from jax.sharding import Sharding, PmapSharding


def sharding_vis(arr, **kwargs):
    if not isinstance(arr, Array):
        raise ValueError(f"is not a jax array, got {type(arr)}")

    def _visualize(sharding):
        return visualize_sharding(arr.shape, sharding, **kwargs)

    inspect_array_sharding(arr, callback=_visualize)


def get_sharded_dims(shape: Sequence[int], sharding: Sharding) -> list[int]:
    device_indices_map = sharding.devices_indices_map(tuple(shape))
    slcs = next(iter(device_indices_map.values()))
    sharded_dims = []
    sl: slice
    for i, sl in enumerate(slcs):
        if sl.start is not None:
            sharded_dims.append(i)
    return sharded_dims


def visualize_sharding(shape: Sequence[int], sharding: Sharding, *,
                       use_color: bool = False, scale: float = 1.,
                       min_width: int = 9, max_width: int = 80,
                       color_map: Optional[ColorMap] = None):
    """
    based on `jax.debug.visualize_array_sharding` and `jax.debug.visualize_sharding`
    """
    console = rich.console.Console(width=max_width)
    use_color = use_color and console.color_system is not None
    if use_color and not color_map:
        try:
            import matplotlib as mpl  # pytype: disable=import-error
            color_map = mpl.colormaps["tab20b"]
        except ModuleNotFoundError:
            use_color = False

    base_height = int(10 * scale)
    aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
    base_width = int(base_height * aspect_ratio)
    height_to_width_ratio = 2.5

    # Grab the device kind from the first device
    device_kind = next(iter(sharding.device_set)).platform.upper()

    device_indices_map = sharding.devices_indices_map(tuple(shape))
    slices: Dict[Tuple[int, ...], Set[int]] = {}
    heights: Dict[Tuple[int, ...], Optional[float]] = {}
    widths: Dict[Tuple[int, ...], float] = {}

    dims = list(range(len(shape)))
    if isinstance(sharding, PmapSharding):
        console.print("[red bold]Output for PmapSharding might be incorrect")
        if len(shape) > 2:
            raise NotImplementedError("can only visualize PmapSharding with shapes with less than 3 dimensions")
    if len(shape) > 2 and not isinstance(sharding, PmapSharding):
        sharded_dims = get_sharded_dims(shape, sharding)
        if len(sharded_dims) > 2:
            raise NotImplementedError(f"can only visualize up to 2 sharded dimension. {sharded_dims} are sharded.")
        chosen_dims = sharded_dims.copy()
        while len(chosen_dims) < 2:
            for i in dims:
                if i not in chosen_dims:
                    chosen_dims.append(i)
                    break
        chosen_dims.sort()
        console.rule(title=f"showing dims {chosen_dims} from original shape {shape}")

    for i, (dev, slcs) in enumerate(device_indices_map.items()):
        assert slcs is not None
        slcs = tuple(map(_raise_to_slice, slcs))
        chunk_idxs = tuple(map(_slice_to_chunk_idx, shape, slcs))

        if slcs is None:
            raise NotImplementedError
        if len(slcs) > 1:
            if len(slcs) > 2:
                slcs = tuple([slcs[i] for i in chosen_dims])
                chunk_idxs = tuple([chunk_idxs[i] for i in chosen_dims])
            vert, horiz = slcs
            vert_size = ((vert.stop - vert.start) if vert.stop is not None
                         else shape[0])
            horiz_size = ((horiz.stop - horiz.start) if horiz.stop is not None
                          else shape[1])
            chunk_height = vert_size / shape[0]
            chunk_width = horiz_size / shape[1]
            heights[chunk_idxs] = chunk_height
            widths[chunk_idxs] = chunk_width
        else:
            # In the 1D case, we set the height to 1.
            horiz, = slcs
            vert = slice(0, 1, None)
            horiz_size = (
                (horiz.stop - horiz.start) if horiz.stop is not None else shape[0])
            chunk_idxs = (0, *chunk_idxs)
            heights[chunk_idxs] = None
            widths[chunk_idxs] = horiz_size / shape[0]
        slices.setdefault(chunk_idxs, set()).add(dev.id)
    num_rows = max([a[0] for a in slices.keys()]) + 1
    if len(list(slices.keys())[0]) == 1:
        num_cols = 1
    else:
        num_cols = max([a[1] for a in slices.keys()]) + 1
    color_iter = make_color_iter(color_map, num_rows, num_cols)
    table = rich.table.Table(show_header=False, show_lines=not use_color,
                             padding=0,
                             highlight=not use_color, pad_edge=False,
                             box=rich.box.SQUARE if not use_color else None)
    for i in range(num_rows):
        col = []
        for j in range(num_cols):
            entry = f"{device_kind} " + ",".join([str(s) for s in sorted(slices[i, j])])
            width, maybe_height = widths[i, j], heights[i, j]
            width = int(width * base_width * height_to_width_ratio)
            if maybe_height is None:
                height = 1
            else:
                height = int(maybe_height * base_height)
            width = min(max(width, min_width), max_width)
            left_padding, remainder = divmod(width - len(entry) - 2, 2)
            right_padding = left_padding + remainder
            top_padding, remainder = divmod(height - 2, 2)
            bottom_padding = top_padding + remainder
            if use_color:
                color = _canonicalize_color(next(color_iter)[:3])
                text_color = _get_text_color(color)
                top_padding += 1
                bottom_padding += 1
                left_padding += 1
                right_padding += 1
            else:
                color = None
                text_color = None
            padding = (top_padding, right_padding, bottom_padding, left_padding)
            padding = tuple(max(x, 0) for x in padding)  # type: ignore
            col.append(
                rich.padding.Padding(
                    rich.align.Align(entry, "center", vertical="middle"), padding,
                    style=rich.style.Style(bgcolor=color,
                                           color=text_color)))
        table.add_row(*col)
    console.print(table, end='\n\n')
