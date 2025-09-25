# This file is part of cpu_cores released under the MIT license.
# See the LICENSE file for more information.

import platform
import sys
import subprocess

from .common import CPUCoresCounter

class WindowsCPUCoresCounter(CPUCoresCounter):

    def _count(self, cpuinfo_filepath=None):
        # powershell is available from Windows 10, wmic is optional from Windows 11 2024
        if sys.getwindowsversion().major >= 10:
            processors = int(
                subprocess.check_output(
                    (
                        'powershell',
                        '-Command',
                        'Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty NumberOfProcessors'
                    ),
                    text=True,
                )
                .strip()
            )

            cores = sum(
                int(core) for core in subprocess.check_output(
                    (
                        'powershell',
                        '-Command',
                        'Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfCores'
                    ),
                    text=True
                )
                .strip()
                .split('\n')
                if core
            )
        else:
            processors = int(
                subprocess.check_output(
                    (
                        'wmic',
                        'computersystem',
                        'get',
                        'NumberOfProcessors',
                        '/value',
                    ),
                    text=True,
                )
                .strip()
                .split('=')[1]
            )

            cores = sum(
                int(line.strip().split('=')[1])
                for line in subprocess.check_output(
                    ('wmic', 'cpu', 'get', 'NumberOfCores', '/value'), text=True
                )
                .strip()
                .split('\n')
                if line.strip()
            )

        if cores == 0 or processors == 0:
            raise Exception("can't get the cpu cores count (windows)")

        self._physical_cores_count = cores
        self._physical_processors_count = processors
