"""Pytest configuration.

This repo uses a simple flat package layout (`core/`, `agents/`, `tools/`) at
the repository root. To make imports reliable in all environments, we add the
project root to `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

