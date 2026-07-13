"""Pytest session bootstrap.

On this Windows setup, importing pandas before torch breaks torch's native DLL
load (WinError 1114 on c10.dll), which segfaults the process when both are later
used together. Importing torch first — before any test module pulls in pandas —
loads torch's DLLs cleanly and avoids the crash. This mirrors the same ordering
rule the runtime scripts follow (see bornil.runtime).
"""
try:  # torch is optional for the numpy-only test subset
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    pass
