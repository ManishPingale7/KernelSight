"""Parse the per-function resource report printed by ``ptxas -v``.

Example input::

    ptxas info    : Compiling entry function 'matmul_tiled' for 'sm_89'
    ptxas info    : Function properties for matmul_tiled
        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 38 registers, used 1 barriers, 2048 bytes smem, 380 bytes cmem[0]

These numbers come from the assembler after register allocation, so they
are the ground truth that PTX-level estimates are compared against:

registers      physical 32-bit registers per thread
spill_stores   bytes written to local memory because registers ran out
spill_loads    bytes read back from those spills
stack_frame    bytes of per-thread local memory (spills and local arrays)
shared_mem     bytes of static shared memory per block
barriers       named barriers used (bar.sync IDs)

Device functions (``.func``) that are compiled together with their callers
only get a "Function properties" line, so their register count is None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_COMPILING = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_PROPERTIES = re.compile(r"Function properties for (\S+)")
_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED = re.compile(r"Used (\d+) registers")
_BARRIERS = re.compile(r"used (\d+) barriers")
_SMEM = re.compile(r"(\d+) bytes smem")
_CMEM = re.compile(r"(\d+) bytes cmem\[(\d+)\]")


@dataclass
class ResourceUsage:
    function: str
    arch: str | None = None
    registers: int | None = None
    barriers: int = 0
    shared_mem: int = 0
    stack_frame: int = 0
    spill_stores: int = 0
    spill_loads: int = 0
    const_mem: dict[int, int] = field(default_factory=dict)  # bank -> bytes

    @property
    def spills(self) -> bool:
        return self.spill_stores > 0 or self.spill_loads > 0


def parse_ptxas_output(text: str) -> dict[str, ResourceUsage]:
    """Map function name -> ResourceUsage."""
    usage: dict[str, ResourceUsage] = {}
    current: ResourceUsage | None = None  # entry function being compiled
    pending: ResourceUsage | None = None  # awaiting its stack-frame line

    def get(name: str) -> ResourceUsage:
        return usage.setdefault(name, ResourceUsage(name))

    for line in text.splitlines():
        if m := _COMPILING.search(line):
            current = get(m.group(1))
            current.arch = m.group(2)
        elif m := _PROPERTIES.search(line):
            pending = get(m.group(1))
        elif (m := _FRAME.search(line)) and pending is not None:
            pending.stack_frame, pending.spill_stores, pending.spill_loads = map(int, m.groups())
            pending = None
        elif (m := _USED.search(line)) and current is not None:
            current.registers = int(m.group(1))
            if b := _BARRIERS.search(line):
                current.barriers = int(b.group(1))
            if s := _SMEM.search(line):
                current.shared_mem = int(s.group(1))
            current.const_mem = {int(bank): int(n) for n, bank in _CMEM.findall(line)}
    return usage
