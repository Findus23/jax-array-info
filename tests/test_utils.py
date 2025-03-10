import ctypes
import ctypes.util
import os
import time


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


if __name__ == '__main__':
    set_process_title("teest")
    print(os.getpid())
    time.sleep(1000)
