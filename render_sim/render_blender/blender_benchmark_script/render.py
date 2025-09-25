import dataclasses as dc
import enum
from typing import List, Optional

import sys
import bpy

import info
from info import DeviceType, Device, ComputeDevice


@dc.dataclass
class RequestedDevice:
    type: DeviceType
    name: Optional[str]


@dc.dataclass
class RenderDevices:
    type: DeviceType
    devices: List[Device]
    cpu_threads: int


def _enable_cpu_device(requested_device: RequestedDevice) -> List[Device]:
    cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
    cycles_preferences.compute_device_type = 'NONE'

    for compute_device in info._get_compute_devices():
        compute_device.cycles_device.use = False

    cpu_devices = [d for d in info._get_all_devices() if d.type == DeviceType.CPU]

    if requested_device.name is not None:
        for device in cpu_devices:
            if device.name != requested_device.name:
                raise ValueError(
                    f"CPU name {device.name} did not match requested CPU name "
                    f"{requested_device.name}, consider omitting the device name"
                )

    return cpu_devices


def _enable_compute_device(requested_device: RequestedDevice) -> List[Device]:
    cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
    cycles_preferences.compute_device_type = requested_device.type.value

    devices = info._get_compute_devices()

    for device in devices:
        device.cycles_device.use = False

    enabled_device: Optional[ComputeDevice]
    if requested_device.name is None:
        non_display_devices_of_correct_type = (
            d for d in devices if d.type == requested_device.type and not d.is_display
        )
        any_device_of_correct_type = (d for d in devices if d.type == requested_device.type)

        enabled_device = next(non_display_devices_of_correct_type, None)
        if enabled_device is None:
            enabled_device = next(any_device_of_correct_type, None)
    else:
        requested_devices = (
            d
            for d in devices
            if d.type == requested_device.type and d.name == requested_device.name
        )

        enabled_device = next(requested_devices, None)

    if enabled_device is not None:
        enabled_device.cycles_device.use = True
        return [enabled_device]
    else:
        return []


def _enable_device(requested_device: RequestedDevice) -> List[Device]:
    if requested_device.type == DeviceType.CPU:
        return _enable_cpu_device(requested_device)
    else:
        return _enable_compute_device(requested_device)


class RenderType(enum.Enum):
    warmup: str = 'warmup'
    full: str = 'full'


def render(render_type: RenderType, requested_device: RequestedDevice) -> RenderDevices:
    scene = bpy.context.scene

    if requested_device.type == DeviceType.CPU:
        scene.cycles.device = 'CPU'
    else:
        scene.cycles.device = 'GPU'

    scene.cycles.use_adaptive_sampling = False

    if render_type == RenderType.warmup:
        # Configure the scene to do minimal amount of render work, just to make
        # the device fully ready to perform actual rendering.
        scene.cycles.samples = 1
    else:
        # Set number of samples to maximum supported value. This ensures that
        # even on very fast device the rendering will happen for an extended
        # period of time.
        scene.cycles.samples = 1 << 24

        scene.cycles.time_limit = 30

    # Communicate configured time limit to the launcher.
    print(f"Time limit is set to {scene.cycles.time_limit}")
    sys.stdout.flush()

    # We are measuring average time per sample, which is not compatible with
    # adaptive sampling.
    scene.cycles.use_adaptive_sampling = False

    # Disable tiling so that we get timing information from an entire frame,
    # even if it happenned to be so that the file is using tiles.
    scene.cycles.use_auto_tile = False

    enabled_devices = _enable_device(requested_device)

    if not enabled_devices:
        raise ValueError(
            f'No matching device of type {requested_device.type.name} found. Options are: '
            f'{[d.name for d in info._get_all_devices() if d.type == requested_device.type]}'
        )

    render_device = RenderDevices(
        type=requested_device.type, devices=enabled_devices, cpu_threads=scene.render.threads
    )

    bpy.ops.render.render()

    return render_device
