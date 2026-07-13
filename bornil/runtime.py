"""Import-order guard for runtime scripts.

On this Windows setup, importing pandas before torch breaks torch's DLL load
(WinError 1114 on c10.dll) and segfaults when both are used together. Any script
that uses BOTH torch and pandas must import torch first. Import this module as
the very first project import to guarantee that ordering:

    import bornil.runtime   # noqa: F401  (loads torch before pandas)
    import pandas as pd
    ...
"""
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - torch optional for numpy-only tooling
    pass
