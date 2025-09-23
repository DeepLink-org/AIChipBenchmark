# This file is part of cpu_cores released under the MIT license.
# See the LICENSE file for more information.

import sys


class CPUCoresCounter(object):

    platform = None
    _physical_cores_count = None
    _physical_processors_count = None

    def _count(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def factory(cls, force_platform=None):
        if force_platform is not None:
            cls.platform = force_platform
        else:
            cls.platform = sys.platform
        if cls.platform.startswith('darwin'):
            from .darwin import DarwinCPUCoresCounter
            return DarwinCPUCoresCounter()
        elif cls.platform.startswith('linux'):
            from .linux import LinuxCPUCoresCounter
            return LinuxCPUCoresCounter()
        elif cls.platform.startswith('win'):
            from .windows import WindowsCPUCoresCounter
            return WindowsCPUCoresCounter()
        else:
            raise NotImplementedError("unsupported platform type [%s]" %
                                      cls.platform)

    def _check_counting_or_do_it(self):
        if self._physical_processors_count is None or \
                self._physical_cores_count is None:
            self._count()

    def get_physical_cores_count(self):
        self._check_counting_or_do_it()
        return self._physical_cores_count

    def get_physical_processors_count(self):
        self._check_counting_or_do_it()
        return self._physical_processors_count
