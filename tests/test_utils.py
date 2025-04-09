import ctypes
import ctypes.util
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from jaxlib import xla_client


def set_process_title(title: str):
    """
    The setproctitle package[1] is the proper portable solution. But I want to have a quick solution
    without dependencies that doesn't need to be portable. So I'll call libc directly instead.

    If it fails, it will return without throwing any error.

    Unlike setproctitle, this only sets /proc/[pid]/comm

    [1]: https://pypi.org/project/setproctitle/
    """
    libc_path = ctypes.util.find_library('c')
    if not libc_path:
        return

    libc = ctypes.CDLL(libc_path, use_errno=True)

    try:
        PR_SET_NAME = getattr(libc, "PR_SET_NAME", 15)
    except AttributeError:
        PR_SET_NAME = 15

    result = libc.prctl(PR_SET_NAME, title.encode())


def todotgraph(hlo_text: str):
    return (
            f"// {datetime.now()}\n" +
            xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(hlo_text))
    )


def save_dot_graph(hlo_text: str, file: Path):
    with file.open("w") as f:
        f.write(todotgraph(hlo_text))
    # with file.with_suffix(".dot").open("w") as f:
    subprocess.run(["dot", "-T", "svg", "-O", str(file)])


def is_on_cluster() -> bool:
    return "SLURM_JOB_ID" in os.environ


def add_xla_flag(flag: str) -> None:
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = flag
        return

    os.environ["XLA_FLAGS"] = os.environ["XLA_FLAGS"] + " " + flag


if __name__ == '__main__':
    set_process_title("teest")
    print(os.getpid())
    time.sleep(1000)
