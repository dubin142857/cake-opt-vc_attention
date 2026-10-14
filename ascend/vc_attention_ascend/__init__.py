"""Experimental Ascend 910B forward attention; see ASCEND_910B.md."""

from .api import attention, prepare, run_prepared, load_native

__version__ = "0.1.0"
__all__ = ["attention", "prepare", "run_prepared", "load_native"]
