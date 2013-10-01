#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
sys.path.append('/home/dsuess/Documents/Diplomarbeit/Hierarchical Equations/')
# This is where settings.py lies
sys.path.append(os.path.abspath('../'))

from settings import PtPerIn, columnwidth, DPI
import numpy as np
from pymol import cmd

from aggregats import AggregatTransferFromFile


###############################################################################
#ratio = 1.6180339887
ratio = .8
width = columnwidth / 2 / PtPerIn * DPI
height = ratio * width


# Scaling function to calculate transparency from proababilty
def p_to_alpha_rel(val):
   maxv = np.max(val)
   return 0.95 * (1 - val / maxv)


def p_to_alpha_abs(val):
   return (1 - val)

#p_to_alpha = p_to_alpha_rel
p_to_alpha = p_to_alpha_rel

# where tosda find the data
fname = '/home/dsuess/Documents/Diplomarbeit/archive/fmo/fmo300-start6-2term-2.pkl'
A = AggregatTransferFromFile(fname)

cmd.set_view ([
    -0.480811268,    0.603387773,    0.636203885,
     0.675508440,    0.717507422,   -0.169974893,
    -0.559037924,    0.348030269,   -0.752567112,
    -0.000127681,    0.000030167, -122.974456787,
    13.921709061,   -7.469791889,   -4.264435768,
  -1086.176879883, 1332.132446289,  -20.000000000])
# select only a few sample points
indices = [0, 500, 1000, 2000, 3999]
t = np.linspace(0, A._tLength, A._tSteps)[indices]

# Get values ordered by time in first index
values = np.swapaxes(A.get(), 0, 1)[indices]

selections = ['lab{}'.format(i + 1) for i in range(8)]
for lab in selections:
   cmd.label(lab, '')

selections = ['bcl{}'.format(i + 1) for i in range(7)]
for i, val in enumerate(values):
   alpha = p_to_alpha(val)
   for n, bcl in enumerate(selections):
      cmd.set_bond('stick_transparency', alpha[n], bcl)

   cmd.refresh()
   cmd.ray(width, height)
   print('Belongs to time {}'.format(t[i] * 5.20883746))
   cmd.png('../fmo_transfer_{}.png'.format(i), dpi=DPI, ray=1)
