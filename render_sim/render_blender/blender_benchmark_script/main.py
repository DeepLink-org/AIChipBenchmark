#!/usr/bin/env python3
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import NewType, Tuple

sys.path.insert(0, str(Path(__file__).parent))
print(f'path: {sys.path}')

import info
import render
from info import DeviceType
from render import RenderType, RequestedDevice

import bpy

ListDevices = NewType('ListDevices', bool)


def parse_arguments() -> Tuple[ListDevices, RenderType, RequestedDevice]:
    parser = argparse.ArgumentParser(prog='main.py', description='Cycles benchmark helper script.')
    parser.add_argument(
        '-l',
        '--list-devices',
        dest='list_devices',
        help='List all devices instead of running the benchmark.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-o',
        '--output',
        dest='output',
        help='Path to save the rendered image (PNG format), can include filename (default filename: background.png).',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-w',
        '--warm-up',
        dest='warmup',
        help='Run a quick warm-up render instead of a full render.',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-t',
        '--device-type',
        dest='device_type',
        help='Device type to be benchmarked (default: CPU).',
        choices=[d.value for d in DeviceType],
        type=str,
        default='CPU',
    )
    parser.add_argument(
        '-d', '--device', dest='device', help='Device to be benchmarked.', type=str, default=None
    )

    argv = sys.argv[sys.argv.index('--') + 1 :]
    parsed_arguments = parser.parse_args(argv)

    return (
        ListDevices(parsed_arguments.list_devices),
        RenderType.warmup if parsed_arguments.warmup else RenderType.full,
        RequestedDevice(
            type=DeviceType(parsed_arguments.device_type),
            name=parsed_arguments.device,
        ),
        parsed_arguments.output
    )


def _list_devices() -> None:
    print(
        'Benchmark JSON Device List: {}'.format(
            json.dumps([d.to_dict() for d in info._get_all_devices()])
        )
    )


def _benchmark(render_type: RenderType, requested_device: RequestedDevice, output = None) -> None:
    render_device_full = render.render(render_type, requested_device)

    print(
        'Benchmark JSON Data: {}'.format(
            json.dumps(
                {
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'blender_version': info.get_blender_version(),
                    'system_info': info.get_system_info(),
                    'device_info': {
                        'device_type': render_device_full.type.value,
                        'compute_devices': [d.to_dict() for d in render_device_full.devices],
                        'num_cpu_threads': render_device_full.cpu_threads,
                    },
                }
            )
        )
    )
    
    if render_type != RenderType.warmup and output:
        if 'Render Result' in bpy.data.images:
            output_path = Path(output)
            if output_path.suffix:
                file_path = output_path
                Path(file_path.parent).mkdir(parents=True, exist_ok=True)
            else:
                Path(output_path).mkdir(parents=True, exist_ok=True)
                file_path = output_path / "background.png"
            render_result = bpy.data.images['Render Result']
            render_result.file_format = 'PNG'
            render_result.save_render(filepath=str(file_path))
            print(f"图像已保存到: {file_path}")


def main() -> None:
    list_devices, render_type, requested_device, output = parse_arguments()

    if list_devices:
        _list_devices()
    else:
        _benchmark(render_type, requested_device, output)


if __name__ == '__main__':
    main()
