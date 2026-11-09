"""Run a command, then print its wall time and peak memory (resident set size)."""
import resource
import subprocess
import sys
import time

t0 = time.time()
code = subprocess.run(sys.argv[1:]).returncode
peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024 / 1e9  # ru_maxrss is in KiB on Linux
print(f"[measure] exit={code} wall={time.time() - t0:.0f}s peak_memory={peak_gb:.2f} GB", flush=True)
sys.exit(code)
