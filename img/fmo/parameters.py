#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
# This is where settings.py lies
sys.path.append(os.path.abspath('../'))

from settings import PtPerIn, columnwidth, DPI


###############################################################################
ratio = 1 / (1.6180339887)
width = columnwidth / PtPerIn * DPI
height = ratio * width
filename = 'fmo_molecule.png'

print('width={}'.format(width))
print('height={}'.format(height))
