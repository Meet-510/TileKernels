"""Configuration for the compile tests, which must not run in interpreter mode.

@triton.jit picks its implementation when a kernel is decorated. If TRITON_INTERPRET is set at
that moment, the kernel becomes an InterpretedFunction, which triton.compile cannot accept. So
these tests cannot share a process with tests/, and tests/test_compile_suite.py runs them in a
separate one.
"""

import os

os.environ.pop("TRITON_INTERPRET", None)
