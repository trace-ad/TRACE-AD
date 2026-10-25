#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[Deprecated] Compatibility shim for the old name. The implementation has moved to core.py.

Run per-series with `<id>.py` (e.g., python 005.py); batch with `python core.py --all`.
"""
from core import *      # noqa: F401,F403
from core import main

if __name__ == "__main__":
    main()
