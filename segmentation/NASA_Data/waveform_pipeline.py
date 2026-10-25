#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[deprecated] compatibility shim for the old name. The implementation has moved to core.py.

Per-series execution is `<id>.py` (e.g. python 005.py); batch is `python core.py --all`.
"""
from core import *      # noqa: F401,F403
from core import main

if __name__ == "__main__":
    main()
