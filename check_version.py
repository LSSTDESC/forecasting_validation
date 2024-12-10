import sys
import numpy as np
import platform
import os

def get_memory_info():
    if sys.platform == "darwin":  # macOS
        mem_bytes = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        mem_gb = mem_bytes / (1024 ** 3)
        return f"{mem_gb:.2f} GB"
    return "Unknown"

def get_os_version():
    if sys.platform == "darwin":  # macOS
        mac_version = platform.mac_ver()[0]
        if not mac_version:  # Fallback if mac_ver() is empty
            mac_version = platform.release()
        return f"macOS {mac_version}"
    return platform.system()

# Gather versions and system info
python_version = sys.version.split()[0]
numpy_version = np.__version__
os_version = get_os_version()
processor = platform.processor()
memory = get_memory_info()

# Print the information
print(f"Python version: {python_version}")
print(f"NumPy version: {numpy_version}")
print(f"Operating System: {os_version}")
print(f"Processor: {processor}")
print(f"Memory: {memory}")
