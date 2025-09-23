# This file is part of cpu_cores released under the MIT license.
# See the LICENSE file for more information.

import shlex
import subprocess

from .common import CPUCoresCounter

CPUINFO_COMMAND = "/usr/sbin/system_profiler" \
                  " -detailLevel full SPHardwareDataType"


class DarwinCPUCoresCounter(CPUCoresCounter):

    def _count(self, command=None):
        if command is None:
            command = CPUINFO_COMMAND
        s = subprocess.Popen(shlex.split(command),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        if s:
            out, err = s.communicate()
            if len(err.strip()) > 0 or len(out.strip()) == 0:
                raise Exception('impossible to get the cpu cores count' +
                                '(darwin) (error message = %s)' % err.strip())
            lines = out.split(b'\n')
            for line in lines:
                tmp = line.strip()
                if tmp.startswith(b'Total Number of Cores:'):
                    self._physical_cores_count = int(tmp.split(b':')[1].split()[0])
                if tmp.startswith(b'Number of Processors:'):
                    self._physical_processors_count = int(tmp.split(b':')[1])
        if self._physical_processors_count is None:
            self._physical_processors_count = 1
        if self._physical_processors_count is None or \
                self._physical_cores_count is None:
            raise Exception('impossible to get the cpu cores count (darwin)')
