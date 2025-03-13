"""
This file runs tests where jax is used in a multiprocess environment.
Therefore this file is only responsible for calling the other processes and the actual jax code is in multihost.py

As this uses the normal `jax.distributed.initialize` setup, this should give identical results
to "real" multihost setups (apart from the fact CPU devices are used).

This is a bit slow as it needs to start N python subprocesses loading jax
for each test. But it makes sure the tests are properly isolated and similar
to a "real" multihost setup.
"""
import subprocess
import sys
from pathlib import Path
from subprocess import Popen

this_file_dir = Path(__file__).parent
multihost_file = this_file_dir / "multihost.py"


def start_multihost(testname: str, num_processes: int):
    """
    Create N subprocesses and pass them the information needed for
    `jax.distributed.initialize` to work.
    """
    procs: list[Popen] = []
    for pindex in range(num_processes):
        proc = Popen(
            [sys.executable, str(multihost_file), testname, str(pindex), str(num_processes)],
            cwd=this_file_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        procs.append(proc)
    return procs


def check_multihost(testname: str, expected_stdout: str, num_processes: int = 4, expect_failure: bool = False):
    """
    compare output and return code to what is expected (for pytest tests)
    """
    procs = start_multihost(testname, num_processes)
    for i, proc in enumerate(procs):
        proc.wait()
        if proc.returncode != 0 and not expect_failure:
            print(proc.stderr.read())
            assert proc.returncode == 0

        if expect_failure:
            assert proc.returncode == 55
        stdout = proc.stdout.read().decode()
        local_expected_stdout = expected_stdout.replace("[IDX]", str(i))
        assert stdout == local_expected_stdout, f"failure at process {i}"


def print_multihost(testname: str, num_processes: int = 4):
    """
    run test and print output directly
    (for use outside of pytest)
    """
    procs = start_multihost(testname, num_processes)
    for i, proc in enumerate(procs):
        proc.wait()
        print("host:", i)
        print("returncode:", proc.returncode)
        print("stdout:")
        stdout = proc.stdout.read().decode()
        print(stdout)
        stderr = proc.stderr.read().decode()
        if stderr:
            print("stderr:")
            print(stderr)


def test_multihost():
    check_multihost("run_empty", "success at [IDX]\n")
    assert True


def test_multihost_device_put():
    expected = """
╭─────────────── simple_array ────────────────╮
│ shape: (32, 32, 32)                         │
│ dtype: complex64                            │
│ size: 256.0 KiB                             │
│ !is_fully_addressable                       │
│ NamedSharding: P(None, 'gpus')              │
│ axis 1 is sharded: CPU 0 contains 0:8 (1/4) │
│                    Total size: 32           │
╰─────────────────────────────────────────────╯
───────────── showing dims [0, 1] from original shape (32, 32, 32) ─────────────
┌───────┬──────────┬──────────┬──────────┐
│       │          │          │          │
│       │          │          │          │
│       │          │          │          │
│       │          │          │          │
│ CPU 0 │CPU 131072│CPU 262144│CPU 393216│
│       │          │          │          │
│       │          │          │          │
│       │          │          │          │
│       │          │          │          │
└───────┴──────────┴──────────┴──────────┘
                            allocated jax arrays                            
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ size     ┃ shape        ┃ dtype     ┃       sharded       ┃ label        ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 64.0 KiB │ (32, 32, 32) │ complex64 │ ✔ (256.0 KiB total) │ simple_array │
├──────────┼──────────────┼───────────┼─────────────────────┼──────────────┤
│ 64.0 KiB │              │           │                     │              │
└──────────┴──────────────┴───────────┴─────────────────────┴──────────────┘
""".lstrip()
    check_multihost("run_multihost_device_put", expected, num_processes=4)
    # I'll assume for now that the CPU ids are deterministic as they are exactly 2^17 apart


def test_multihost_closure():
    expected_error = (
        "Closing over jax.Array that spans non-addressable (non process local) devices is not allowed. "
        "Please pass such arrays as arguments to the function. "
        "Got jax.Array: float32[16,16]\n"
    )
    check_multihost("run_multihost_closure", expected_error, expect_failure=True)


def test_multihost_local_shard():
    expected_output = """
Fetching value for `jax.Array` that spans non-addressable (non process local) devices is not possible. You can use `jax.experimental.multihost_utils.process_allgather` to print the global array or use `.addressable_shards` method of jax.Array to inspect the addressable (process local) shards.
╭─ local_subset_np ──╮
│ shape: (32, 8, 32) │
│ dtype: complex64   │
│ size: 64.0 KiB     │
│ numpy array        │
╰────────────────────╯
""".lstrip()
    check_multihost("run_multihost_local_shard", expected_output)


def test_numpy_to_sharded_array():
    expected_output = """
╭───────────── distributed_array ──────────────╮
│ shape: (128, 128)                            │
│ dtype: float32                               │
│ size: 64.0 KiB                               │
│ !is_fully_addressable                        │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:32 (1/4) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
""".lstrip()
    check_multihost("run_numpy_to_sharded_array", expected_output)


def test_host_local_array_to_global_array():
    expected_output = """
╭──────────────── global_array ────────────────╮
│ shape: (100,)                                │
│ dtype: int32                                 │
│ size: 400.0 B                                │
│ !is_fully_addressable                        │
│ NamedSharding: P('gpus',)                    │
│ axis 0 is sharded: CPU 0 contains 0:25 (1/4) │
│                    Total size: 100           │
╰──────────────────────────────────────────────╯
""".lstrip()
    check_multihost("run_host_local_array_to_global_array", expected_output)


def test_process_allgather():
    expected_output = """
╭───── arr_np ──────╮
│ shape: (128, 128) │
│ dtype: float32    │
│ size: 64.0 KiB    │
│ numpy array       │
╰───────────────────╯
""".lstrip()
    check_multihost("run_process_allgather", expected_output)


def test_shard_map():
    expected_output = """
╭─ x (in shard_map) ─╮
│ shape: (128, 32)   │
│ dtype: float32     │
│ size: 16.0 KiB     │
│ called in jit      │
╰────────────────────╯
╭───────── out ─────────╮
│ shape: (128, 32)      │
│ dtype: float32        │
│ size: 16.0 KiB        │
│ !is_fully_addressable │
│ NamedSharding: P()    │
╰───────────────────────╯
╭─ x (in shard_map) ─╮
│ shape: (128, 32)   │
│ dtype: float32     │
│ size: 16.0 KiB     │
│ called in jit      │
╰────────────────────╯
""".lstrip()
    check_multihost("run_shard_map", expected_output)


def test_broadcast_one_to_all():
    check_multihost("run_broadcast_one_to_all", "")


def test_custom_rfftn_multigpu():
    expected_output = """
╭──────────────── input_array ─────────────────╮
│ shape: (128, 128, 128)                       │
│ dtype: float32                               │
│ size: 8.0 MiB                                │
│ !is_fully_addressable                        │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
╭──────────────── output_array ────────────────╮
│ shape: (128, 128, 65)                        │
│ dtype: complex64                             │
│ size: 8.1 MiB                                │
│ !is_fully_addressable                        │
│ NamedSharding: P(None, 'gpus')               │
│ axis 1 is sharded: CPU 0 contains 0:16 (1/8) │
│                    Total size: 128           │
╰──────────────────────────────────────────────╯
""".lstrip()
    check_multihost("run_custom_rfftn_multigpu", expected_output, num_processes=8)


if __name__ == '__main__':
    test = sys.argv[1]
    print_multihost(test)
