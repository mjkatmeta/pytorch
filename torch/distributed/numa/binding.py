import functools
import os
import shutil
import subprocess
import traceback
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, overload, TypeVar, Union

import torch
from torch._utils_internal import signpost_event
from torch.distributed.elastic.utils.logging import get_logger


__all__ = [
    "maybe_wrap_entrypoint_with_numa_bindings",
    "AffinityMode",
    "NumaOptions",
]

_NUMACTL_COMMAND = "numactl"

logger = get_logger(__file__)


class AffinityMode(str, Enum):
    """
    See behavior description for each affinity mode
    in torch.distributed.run.
    """

    NODE = "node"
    SOCKET = "socket"
    EXCLUSIVE = "exclusive"
    CORE_COMPLEX = "core-complex"


@dataclass(frozen=True)
class NumaOptions:
    affinity_mode: AffinityMode

    """
    If true, we will fall back to using the original command/entrypoint if we fail to compute
    or apply NUMA bindings.

    You should avoid using this option! It is only intended as a safety mechanism for facilitating
    mass rollouts of numa binding.
    """
    should_fall_back_if_binding_fails: bool = False


@overload
def maybe_wrap_entrypoint_with_numa_bindings(
    *,
    entrypoint: tuple[str, ...],
    gpu_index: int,
    numa_options: Optional[NumaOptions],
) -> tuple[str, ...]: ...


@overload
def maybe_wrap_entrypoint_with_numa_bindings(
    *,
    entrypoint: Callable,
    gpu_index: int,
    numa_options: Optional[NumaOptions],
) -> Callable: ...


def maybe_wrap_entrypoint_with_numa_bindings(
    *,
    entrypoint: Union[tuple[str, ...], Callable],
    gpu_index: int,
    numa_options: Optional[NumaOptions],
) -> Union[tuple[str, ...], Callable]:
    if numa_options is None:
        return entrypoint

    try:
        logical_cpu_indices = _get_logical_cpu_indices_to_bind_to(
            gpu_index=gpu_index, numa_options=numa_options
        )
        logger.info(
            "Will attempt to bind to logical_cpu_indices=%r", logical_cpu_indices
        )

        wrapped: Union[tuple[str, ...], Callable]
        if callable(entrypoint):
            wrapped = _bind_callable_to_logical_cpus(
                _callable=entrypoint,
                logical_cpu_indices=logical_cpu_indices,
                numa_options=numa_options,
            )
        else:
            wrapped = _bind_command_args_to_logical_cpus_and_validate(
                command_args=entrypoint, logical_cpu_indices=logical_cpu_indices
            )

        logger.info("NUMA binding succeeded!")
        signpost_event(
            category="numa_binding",
            name="binding_success",
            parameters={"logical_cpu_indices": logical_cpu_indices},
        )

        return wrapped
    except Exception:
        traceback_str = traceback.format_exc()
        signpost_event(
            category="numa_binding",
            name="binding_exception",
            parameters={
                "gpu_index": gpu_index,
                "numa_options": numa_options,
                "traceback": traceback_str,
            },
        )
        logger.exception(
            """Failed to determine proper NUMA bindings, with arguments:
                gpu_index=%d,
                numa_options=%r,
                traceback=%r,
        """,
            gpu_index,
            numa_options,
            traceback_str,
        )
        if numa_options.should_fall_back_if_binding_fails:
            logger.warning(
                "Falling back to original command after NUMA binding failed because fallback is enabled."
            )
            return entrypoint
        raise


def _bind_callable_to_logical_cpus(
    *,
    _callable: Callable,
    logical_cpu_indices: set[int],
    numa_options: NumaOptions,
) -> Callable:
    # Use functools.partial with module-level function for picklability
    wrapped_callable = functools.partial(
        _bind_to_logical_cpus_then_invoke_entrypoint,
        logical_cpu_indices,
        _callable,
        numa_options,
    )
    # Copy original callable's metadata
    return functools.update_wrapper(wrapped_callable, _callable)


TEntrypointReturn = TypeVar("TEntrypointReturn")


def _bind_to_logical_cpus_then_invoke_entrypoint(
    logical_cpu_indices: set[int],
    entrypoint: Callable[..., TEntrypointReturn],
    numa_options: NumaOptions,
    *args: object,
    **kwargs: object,
) -> TEntrypointReturn:
    try:
        os.sched_setaffinity(0, logical_cpu_indices)
        logger.info("Successfully applied NUMA bindings for callable entrypoint.")
    except Exception:
        traceback_str = traceback.format_exc()
        logger.exception(
            """Failed to apply NUMA bindings for callable entrypoint, with
            logical_cpu_indices=%r,
            traceback=%s
        """,
            logical_cpu_indices,
            traceback_str,
        )
        signpost_event(
            category="numa_binding",
            name="apply_exception",
            parameters={
                "logical_cpu_indices": logical_cpu_indices,
                "traceback": traceback_str,
            },
        )
        if not numa_options.should_fall_back_if_binding_fails:
            raise
        logger.warning(
            "Falling back to original entrypoint after failing to apply NUMA bindings"
        )

    return entrypoint(*args, **kwargs)


def _bind_command_args_to_logical_cpus_and_validate(
    command_args: tuple[str, ...], logical_cpu_indices: set[int]
) -> tuple[str, ...]:
    # Validate
    _raise_if_numactl_cli_not_available()
    _raise_if_numactl_cli_dry_run_fails(logical_cpu_indices=logical_cpu_indices)

    return _get_numactl_cli_args(
        logical_cpu_indices=logical_cpu_indices, original_command_args=command_args
    )


def _get_logical_cpu_indices_to_bind_to(
    *, gpu_index: int, numa_options: NumaOptions
) -> set[int]:
    if numa_options.affinity_mode == AffinityMode.NODE:
        return _get_logical_cpu_indices_to_bind_to_for_node_affinity(
            gpu_index=gpu_index
        )
    if numa_options.affinity_mode == AffinityMode.SOCKET:
        return _get_logical_cpu_indices_to_bind_to_for_socket_affinity(
            gpu_index=gpu_index
        )
    if numa_options.affinity_mode == AffinityMode.EXCLUSIVE:
        return _get_logical_cpu_indices_to_bind_to_for_exclusive_affinity(
            gpu_index=gpu_index
        )
    if numa_options.affinity_mode == AffinityMode.CORE_COMPLEX:
        return _get_logical_cpu_indices_to_bind_to_for_core_complex_affinity(
            gpu_index=gpu_index
        )
    raise ValueError(f"Affinity mode {numa_options.affinity_mode} not supported.")


def _raise_if_numactl_cli_dry_run_fails(*, logical_cpu_indices: set[int]) -> None:
    noop_args = _get_numactl_cli_args(
        logical_cpu_indices=logical_cpu_indices,
        # Execute arbitrary noop
        original_command_args=("true",),
    )
    try:
        subprocess.run(
            noop_args,
            stdout=subprocess.DEVNULL,
            # These allow us to capture the stderr as text
            stderr=subprocess.PIPE,
            text=True,
            # Raise exception if nonzero exit status.
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"""Our binding logic inferred to prepend your command with options {noop_args[:-1]}.
            Before doing that, we did a noop dry run with args {noop_args}, but that command failed.
            This should NOT happen, and likely suggests a bug in pytorch's numa binding logic.

            The {_NUMACTL_COMMAND} command itself had this stderr:

            {e.stderr}
            """
        ) from e


def _get_numactl_cli_args(
    *, logical_cpu_indices: set[int], original_command_args: tuple[str, ...]
) -> tuple[str, ...]:
    # Syntax for invoking a command but with numactl activated is numactl <args> command <args>
    logical_cpus_str = _get_ranges_str_from_ints(logical_cpu_indices)
    return (
        _NUMACTL_COMMAND,
        f"--physcpubind={logical_cpus_str}",
        *original_command_args,
    )


def _raise_if_numactl_cli_not_available() -> None:
    if not shutil.which(_NUMACTL_COMMAND):
        raise RuntimeError(
            f"{_NUMACTL_COMMAND} shell command is required for NUMA bindings."
        )


def _get_logical_cpu_indices_to_bind_to_for_node_affinity(
    *, gpu_index: int
) -> set[int]:
    """
    Core logic of 'node' numa strategy.

    Returns logical CPU indices to bind to for the given GPU.
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)
    return _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )


def _get_logical_cpu_indices_to_bind_to_for_socket_affinity(
    *, gpu_index: int
) -> set[int]:
    """
    Core logic of 'socket' numa strategy.
    """
    numa_node_index_of_gpu = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)
    socket_index = _get_socket_index_for_numa_node(
        numa_node_index=numa_node_index_of_gpu
    )
    numa_node_indices = _get_numa_node_indices_for_socket_index(
        socket_index=socket_index
    )

    logical_cpu_indices = set()
    for numa_node_index in numa_node_indices:
        logical_cpu_indices.update(
            _get_allowed_logical_cpu_indices_for_numa_node(
                numa_node_index=numa_node_index
            )
        )

    return logical_cpu_indices


def _get_logical_cpu_indices_to_bind_to_for_exclusive_affinity(
    *, gpu_index: int
) -> set[int]:
    """
    Core logic of 'exclusive' numa strategy.
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)

    gpu_indices = _get_gpu_indices_for_numa_node(numa_node_index=numa_node_index)
    gpu_indices = sorted(gpu_indices)
    original_gpu_relative_index = gpu_indices.index(gpu_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the physical core to
    # represent the physical core.
    physical_core_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpu_indices_sharing_same_physical_core_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )
    # Sort the dict for consistency (dicts maintain order in Python)
    physical_core_to_allowed_logical_cpu_indices = dict(
        sorted(physical_core_to_allowed_logical_cpu_indices.items())
    )

    num_physical_cores_per_gpu = len(
        physical_core_to_allowed_logical_cpu_indices
    ) // len(gpu_indices)
    # Often, the number of physical cores will not be perfectly divisible by the number
    # of GPUs. In those cases, give the lowest GPU indices an extra core
    num_gpus_to_give_one_extra_physical_core = len(
        physical_core_to_allowed_logical_cpu_indices
    ) % len(gpu_indices)

    if num_physical_cores_per_gpu < 1:
        raise RuntimeError(
            f"There are only {len(physical_core_to_allowed_logical_cpu_indices)} physical cores on {numa_node_index=},"
            + f" but there are {len(gpu_indices)} GPUs associated with this NUMA node."
        )

    # Compute slice indices for this GPU
    start = original_gpu_relative_index * num_physical_cores_per_gpu + min(
        original_gpu_relative_index, num_gpus_to_give_one_extra_physical_core
    )
    end = (
        start
        + num_physical_cores_per_gpu
        + (
            1
            if original_gpu_relative_index < num_gpus_to_give_one_extra_physical_core
            else 0
        )
    )

    # Slice and flatten the logical CPUs from the selected physical cores
    logical_cpu_indices_for_original_gpu = set()
    for logical_cpu_indices in list(
        physical_core_to_allowed_logical_cpu_indices.values()
    )[start:end]:
        logical_cpu_indices_for_original_gpu.update(logical_cpu_indices)

    return logical_cpu_indices_for_original_gpu


def _get_logical_cpu_indices_to_bind_to_for_core_complex_affinity(
    *, gpu_index: int
) -> set[int]:
    """
    Core logic of 'core-complex' numa strategy.

    Each GPU is assigned a full core complex (group of cores sharing L3 cache)
    within its affined NUMA node.
    """
    numa_node_index = _get_numa_node_index_for_gpu_index(gpu_index=gpu_index)

    gpu_indices = _get_gpu_indices_for_numa_node(numa_node_index=numa_node_index)
    gpu_indices = sorted(gpu_indices)
    original_gpu_relative_index = gpu_indices.index(gpu_index)

    allowed_logical_cpu_indices = _get_allowed_logical_cpu_indices_for_numa_node(
        numa_node_index=numa_node_index
    )

    # Arbitrarily use the min logical cpu index on the max level cache
    # to represent the max level cache.
    max_level_cache_to_allowed_logical_cpu_indices = _group_by(
        allowed_logical_cpu_indices,
        lambda logical_cpu_index: min(
            _get_logical_cpus_sharing_same_max_level_cache_as(
                logical_cpu_index=logical_cpu_index
            )
        ),
    )

    max_level_cache_to_allowed_logical_cpu_indices = dict(
        sorted(
            max_level_cache_to_allowed_logical_cpu_indices.items(),
            # First, prioritize caches with more available cpus
            # Second, prioritize lower index cpus (just for clarity/consistency)
            key=lambda item: (-len(item[1]), item[0]),
        )
    )

    cache_index_for_original_gpu = original_gpu_relative_index % len(
        max_level_cache_to_allowed_logical_cpu_indices
    )
    logical_cpu_indices_for_original_gpu = list(
        max_level_cache_to_allowed_logical_cpu_indices.values()
    )[cache_index_for_original_gpu]

    return logical_cpu_indices_for_original_gpu


K = TypeVar("K")
V = TypeVar("V")


def _group_by(values: Iterable[V], get_key: Callable[[V], K]) -> dict[K, set[V]]:
    """
    Groups elements with same key into sets.
    """
    key_to_values: defaultdict[K, set[V]] = defaultdict(set)
    for value in values:
        key = get_key(value)
        key_to_values[key].add(value)
    return key_to_values


def _get_logical_cpu_indices_sharing_same_physical_core_as(
    *, logical_cpu_index: int
) -> set[int]:
    thread_siblings_list_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/topology/thread_siblings_list"
    )
    with open(thread_siblings_list_absolute_path) as f:
        return _get_set_of_int_from_ranges_str(f.read())


def _get_logical_cpus_sharing_same_max_level_cache_as(
    *, logical_cpu_index: int
) -> set[int]:
    cpu_cache_dir_absolute_path = (
        f"/sys/devices/system/cpu/cpu{logical_cpu_index}/cache"
    )

    max_level = -1
    logical_cpus_sharing_max_level_cache = set()
    for entry in os.listdir(cpu_cache_dir_absolute_path):
        if not entry.startswith("index") or not entry[5:].isdecimal():
            continue
        cache_index_absolute_path = os.path.join(cpu_cache_dir_absolute_path, entry)

        # Filter out other cache types like Instruction
        type_absolute_path = os.path.join(cache_index_absolute_path, "type")
        with open(type_absolute_path) as type_file:
            if type_file.read().strip() not in {"Unified", "Data"}:
                continue

        level_absolute_path = os.path.join(cache_index_absolute_path, "level")
        with open(level_absolute_path) as level_file:
            level = int(level_file.read())
        if level <= max_level:
            continue

        max_level = level
        shared_cpu_list_absolute_path = os.path.join(
            cache_index_absolute_path, "shared_cpu_list"
        )
        with open(shared_cpu_list_absolute_path) as share_cpu_list_file:
            logical_cpus_sharing_max_level_cache = _get_set_of_int_from_ranges_str(
                share_cpu_list_file.read()
            )

    return logical_cpus_sharing_max_level_cache


def _get_allowed_logical_cpu_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    all_cpu_indices = _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
        numa_node_index=numa_node_index
    )
    allowed_cpu_indices = _get_allowed_cpu_indices_for_current_process()
    return all_cpu_indices & allowed_cpu_indices


def _get_cpu_indices_for_numa_node_MAYBE_NOT_ALLOWED(
    *, numa_node_index: int
) -> set[int]:
    """
    Returns:
        Indices of all CPUs associated with numa_node_index. However, the list
        is not filtered based on whether the process is allowed to use them.
    """
    cpulist_absolute_path = f"/sys/devices/system/node/node{numa_node_index}/cpulist"
    try:
        with open(cpulist_absolute_path) as f:
            cpu_range_str = f.read()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not determine CPUs corresponding to {numa_node_index=}."
        ) from e
    return _get_set_of_int_from_ranges_str(cpu_range_str)


def _get_gpu_count() -> int:
    return torch.cuda.device_count()


def _get_numa_node_index_for_gpu_index(*, gpu_index: int) -> int:
    device_properties = torch.cuda.get_device_properties(gpu_index)

    domain = device_properties.pci_domain_id  # type: ignore[attr-defined]
    bus = device_properties.pci_bus_id  # type: ignore[attr-defined]
    device = device_properties.pci_device_id  # type: ignore[attr-defined]

    # Format to sysfs PCI address: "0000:dc:00.0"
    pci_addr = f"{domain:04x}:{bus:02x}:{device:02x}.0"

    pci_numa_node_absolute_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    with open(pci_numa_node_absolute_path) as f:
        # In systems with only one NUMA node, this will
        # often be saved as -1. In those cases, there is obviously
        # at least one numa node, 0, so we use that.
        return max(int(f.read().strip()), 0)


def _get_gpu_indices_for_numa_node(*, numa_node_index: int) -> set[int]:
    return {
        gpu_index
        for gpu_index in range(_get_gpu_count())
        if _get_numa_node_index_for_gpu_index(gpu_index=gpu_index) == numa_node_index
    }


def _get_socket_index_for_numa_node(*, numa_node_index: int) -> int:
    arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
        numa_node_index=numa_node_index
    )

    return _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index)


def _get_socket_index_for_cpu(*, cpu_index: int) -> int:
    package_id_absolute_path = (
        f"/sys/devices/system/cpu/cpu{cpu_index}/topology/physical_package_id"
    )
    try:
        with open(package_id_absolute_path) as f:
            return int(f.read().strip())
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not determine socket for {cpu_index=}") from e


def _get_arbitrary_allowed_cpu_index_for_numa_node(*, numa_node_index: int) -> int:
    return min(
        _get_allowed_logical_cpu_indices_for_numa_node(numa_node_index=numa_node_index)
    )


def _get_set_of_int_from_ranges_str(ranges_str: str) -> set[int]:
    """
    Util for parsing a string of int ranges, as in a sysfs file.

    Args:
        ranges_str: E.g., "0-2,4,6-7"

    Returns:
        E.g., {0, 1, 2, 4, 6, 7}
    """
    ints: set[int] = set()
    for range_str in ranges_str.split(","):
        range_str = range_str.strip()
        if not range_str:
            continue
        if "-" in range_str:
            start_str, end_str = range_str.split("-")
            start, end = int(start_str), int(end_str)
            ints.update(range(start, end + 1))
        else:
            ints.add(int(range_str))
    return ints


def _get_ranges_str_from_ints(ints: Iterable[int]) -> str:
    """
    Convert a set of integers to a compact string with ranges.

    Args:
        ints: E.g., {0, 1, 2, 4, 6, 7}

    Returns:
        E.g., "0-2,4,6-7"
    """
    if not ints:
        return ""

    sorted_ints = sorted(ints)
    ranges = []
    start = prev = sorted_ints[0]

    for num in sorted_ints[1:]:
        if num == prev + 1:
            prev = num
        else:
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = num

    # Append the last range
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def _get_systemwide_numa_node_indices() -> set[int]:
    with open("/sys/devices/system/node/possible") as f:
        possible_nodes_str = f.read()

    return _get_set_of_int_from_ranges_str(possible_nodes_str)


def _get_numa_node_indices_for_socket_index(*, socket_index: int) -> set[int]:
    systemwide_numa_node_indices = _get_systemwide_numa_node_indices()

    matching_numa_node_indices = set()
    for numa_node_index in systemwide_numa_node_indices:
        arbitrary_cpu_index = _get_arbitrary_allowed_cpu_index_for_numa_node(
            numa_node_index=numa_node_index
        )
        if socket_index == _get_socket_index_for_cpu(cpu_index=arbitrary_cpu_index):
            matching_numa_node_indices.add(numa_node_index)

    return matching_numa_node_indices


def _get_allowed_cpu_indices_for_current_process() -> set[int]:
    # 0 denotes current process
    return os.sched_getaffinity(0)
