import dataclasses as dc
import enum
import multiprocessing
import platform
import subprocess
import sys
from typing import Generator, Optional, Dict, List, Any

import _cycles
import bpy

from vendor import distro, cpu_cores


# NOTE: Keep in the order of preference.
# This means that if device supports multiple compute backends only
# the first one will be used. This is because, for example, it is not
# so interesting to check CUDA performance on OptiX-capable device.
class DeviceType(enum.Enum):
    CPU: str = 'CPU'
    HIP: str = 'HIP'
    OptiX: str = 'OPTIX'
    CUDA: str = 'CUDA'
    Metal: str = 'METAL'
    OneAPI: str = 'ONEAPI'


@dc.dataclass
class Device:
    name: str
    type: DeviceType
    is_display: bool

    def to_dict(self) -> Dict[str, object]:
        if self.type == DeviceType.CPU:
            return {'name': self.name, 'type': self.type.value}
        else:
            return {
                'name': self.name,
                'type': self.type.value,
                'is_display': self.is_display,
            }


@dc.dataclass
class ComputeDevice(Device):
    cycles_device: Any


def _get_devices_for_type(type: DeviceType) -> Generator[Any, None, None]:

    try:
        available_devices = _cycles.available_devices(type.value)
    except ValueError:
        # Ignore compute device type which is not supported by the current Cycles version.
        return

    for device in available_devices:
        # Device is a non-strictly-typed tuple. Element with index 1 is the type
        # of the device.
        if device[1] != type.value:
            continue
        yield device


def _get_all_devices() -> List[Device]:
    all_devices = []

    used_devices = []

    for type in DeviceType:
        for name, *_ in _get_devices_for_type(type):
            if name in used_devices:
                continue
            used_devices.append(name)

            all_devices.append(
                Device(
                    name=name.replace(' (Display)', ''),
                    type=DeviceType(type),
                    is_display='(Display)' in name,
                )
            )

    return all_devices


def _get_compute_devices() -> List[ComputeDevice]:
    compute_devices = []

    cycles = bpy.context.preferences.addons['cycles']
    cpref = cycles.preferences

    # Refresh the list so that the Cycles addons caches all devices on its side.
    # Without doing so there is a risk that the copute device pointer will
    # change after requesting a subsequent device type.
    cpref.refresh_devices()

    for type in DeviceType:
        try:
            devices = cpref.get_devices_for_type(type.value)
        except ValueError:
            # Ignore compute device type which is not supported by the current Cycles version.
            continue

        for device in devices:
            compute_devices.append(
                ComputeDevice(
                    name=device.name.replace(' (Display)', ''),
                    type=type,
                    is_display='(Display)' in device.name,
                    cycles_device=device,
                )
            )

    return compute_devices


@dc.dataclass
class CPUTopology:
    sockets: int
    cores: int
    threads: int


def _get_cpu_topology() -> CPUTopology:
    """
    Get topology information (number of sockets, physical and logical threads)
    of the system CPUs.
    """
    cores_info = cpu_cores.CPUCoresCounter.factory()  # type: ignore
    sockets: int = cores_info.get_physical_processors_count()
    cores: int = cores_info.get_physical_cores_count()

    return CPUTopology(sockets=sockets, cores=cores, threads=multiprocessing.cpu_count())


def get_system_info() -> Dict[str, object]:
    system: str = platform.system()

    dist_name: Optional[str] = None
    dist_version: Optional[str] = None
    if system == 'Linux':
        dist_name, dist_version, *_ = distro.linux_distribution()  # type: ignore

    cpu_topology = _get_cpu_topology()
    return {
        'bitness': platform.architecture()[0],
        'machine': platform.machine(),
        'system': platform.system(),
        'dist_name': dist_name,
        'dist_version': dist_version,
        'devices': [d.to_dict() for d in _get_all_devices()],
        'num_cpu_sockets': cpu_topology.sockets,
        'num_cpu_cores': cpu_topology.cores,
        'num_cpu_threads': cpu_topology.threads,
    }


def get_blender_version() -> Dict[str, object]:
    return {
        'version': bpy.app.version_string,
        'build_date': bpy.app.build_date.decode('utf-8'),
        'build_time': bpy.app.build_time.decode('utf-8'),
        'build_commit_date': bpy.app.build_commit_date.decode('utf-8'),
        'build_commit_time': bpy.app.build_commit_time.decode('utf-8'),
        'build_hash': bpy.app.build_hash.decode('utf-8'),
    }
