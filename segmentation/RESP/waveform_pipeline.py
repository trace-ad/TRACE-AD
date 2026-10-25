#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[Deprecated] Compatibility shim for the old name. The implementation has moved to core.py.

For per-series runs use `<id>.py` (e.g. python 024.py); for batch runs use `python core.py --all`.
"""
from core import *      # noqa: F401,F403
from core import main

if __name__ == "__main__":
    main()
