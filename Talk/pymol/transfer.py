#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
sys.path.append('/home/dsuess/Documents/Diplomarbeit/Hierarchical Equations/')

import numpy as np
from pymol import cmd

from aggregats import AggregatTransferFromFile


###############################################################################
width = 1000
height = int(1.3 * width)
sample_points = 250
fname = '/data11/archive/fmo/fmofree-start6.pkl'
# fname = '/data11/archive/fmo/fmo77-start6-2.pkl'
savename = 'free/frame'

# Scaling function to calculate transparency from proababilty
def p_to_alpha_rel(val):
   maxv = np.max(val)
   return 0.95 * (1 - val / maxv)


def p_to_alpha_abs(val):
   return .9 * (1 - val)

p_to_alpha = p_to_alpha_abs
p_to_alpha = p_to_alpha_rel

# where tosda find the data
A = AggregatTransferFromFile(fname)

cmd.set_view ([
    -0.566527486,    0.549390495,    0.614174008,\
     0.489665151,    0.823900998,   -0.285317212,\
    -0.662771285,    0.139101312,   -0.735783637,\
     0.000076182,    0.000004612, -127.713699341,\
    13.776422501,   -7.558099747,   -3.967736244,\
  -413.703674316,  669.118835449,  -20.000000000 ])
# select only a few sample points
indices = np.array(np.linspace(0, A._tSteps, sample_points, endpoint=False), dtype=int)
t = np.linspace(0, A._tLength, A._tSteps)[indices]

# Get values ordered by time in first index
values = np.swapaxes(A.get(), 0, 1)[indices]

selections = ['bcl{}'.format(i + 1) for i in range(7)]
for i, val in enumerate(values):
   print('Rendering {}/{}'.format(i, sample_points))
   alpha = p_to_alpha(val)
   for n, bcl in enumerate(selections):
      cmd.set_bond('stick_transparency', alpha[n], bcl)

   cmd.label('timestamp', '"t = {0:.2f} ps"'.format(t[i] * 5.3088))
   cmd.show('labels', 'timestamp')
   cmd.refresh()
   cmd.ray(width, height)
   cmd.png(savename + "%05d.png" % i)
