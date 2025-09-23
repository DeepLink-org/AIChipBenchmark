# This file is part of cpu_cores released under the MIT license.
# See the LICENSE file for more information.

from .common import CPUCoresCounter

CPUINFO_FILEPATH = "/proc/cpuinfo"


def _core_hash(cpu_infos):
    if 'core id' not in cpu_infos and 'physical id' not in cpu_infos:
        return "%i" % cpu_infos['processor']
    if 'core id' in cpu_infos and 'physical id' not in cpu_infos:
        raise Exception("incorrect cpuinfo file :"
                        " we have a core_id without physical_id")
    if 'core id' in cpu_infos:
        return "%i_%i" % (cpu_infos['physical id'], cpu_infos['core id'])
    else:
        return "%i" % cpu_infos['physical id']


def _processor_hash(cpu_infos):
    if 'core id' not in cpu_infos and 'physical id' not in cpu_infos:
        return "%i" % cpu_infos['processor']
    if 'core id' in cpu_infos and 'physical id' not in cpu_infos:
        raise Exception("incorrect cpuinfo file :"
                        " we have a core_id without physical_id")
    return "%i" % cpu_infos['physical id']


class LinuxCPUCoresCounter(CPUCoresCounter):

    def _count(self, cpuinfo_filepath=None):
        if cpuinfo_filepath is None:
            cpuinfo_filepath = CPUINFO_FILEPATH
        with open(cpuinfo_filepath, 'r') as f:
            # we read lines in reversed order to be sure to end with a
            # "processor:" line
            lines = reversed(f.readlines())
            cores = set()
            processors = set()
            cpu_infos = {}
            for line in lines:
                tmp = line.strip()
                for key in ('processor', 'physical id', 'core id'):
                    if tmp.startswith(key):
                        cpu_infos[key] = int(tmp.split(':')[1].strip())
                        if key == 'processor':
                            cores.add(_core_hash(cpu_infos))
                            processors.add(_processor_hash(cpu_infos))
                            cpu_infos = {}
            if len(cores) == 0 or len(processors) == 0:
                raise Exception("can't get the cpu cores count (linux)")
        self._physical_cores_count = len(cores)
        self._physical_processors_count = len(processors)
