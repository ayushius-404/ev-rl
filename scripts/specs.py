"""Print (and save) the machine and software specs for the report.

    python scripts/specs.py

Works on Windows, Linux and macOS. Writes results/specs.txt as well.
"""
import ctypes
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"])
        if out:
            return out
    elif system == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or "unknown"


def physical_cores():
    try:
        import psutil  # optional
        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
        if out.isdigit():
            return int(out)
    return None


def ram_gb():
    system = platform.system()
    try:
        if system == "Windows":
            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024 ** 3
        if system == "Linux":
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 ** 2
        if system == "Darwin":
            return int(_run(["sysctl", "-n", "hw.memsize"])) / 1024 ** 3
    except Exception:
        pass
    return None


def pkg_version(name: str) -> str:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return "not installed"


def main():
    lines = []
    add = lines.append
    add("=== Machine ===")
    add(f"OS            : {platform.system()} {platform.release()} ({platform.version()})")
    add(f"CPU           : {cpu_name()}")
    pc, lc = physical_cores(), os.cpu_count()
    add(f"CPU cores     : {pc if pc else 'unknown'} physical, {lc} logical")
    r = ram_gb()
    add(f"RAM           : {r:.1f} GB" if r else "RAM           : unknown")

    add("")
    add("=== Software ===")
    add(f"Python        : {sys.version.split()[0]}")
    for label, mod in [("PyTorch", "torch"), ("Stable-Baselines3", "stable_baselines3"),
                       ("Gymnasium", "gymnasium"), ("NumPy", "numpy"), ("pandas", "pandas"),
                       ("CVXPY", "cvxpy"), ("Matplotlib", "matplotlib"), ("Streamlit", "streamlit")]:
        add(f"{label:<14}: {pkg_version(mod)}")

    add("")
    add("=== Compute device used by PyTorch ===")
    try:
        import torch
        cuda = torch.cuda.is_available()
        add(f"CUDA available: {cuda}")
        if cuda:
            add(f"GPU           : {torch.cuda.get_device_name(0)}")
        add(f"Torch threads : {torch.get_num_threads()}")
        add("Training was run on: " + ("GPU" if cuda else "CPU"))
    except Exception:
        add("PyTorch not available")

    text = "\n".join(lines)
    print(text)
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "specs.txt").write_text(text + "\n", encoding="utf-8")
    print("\nSaved results/specs.txt")


if __name__ == "__main__":
    main()