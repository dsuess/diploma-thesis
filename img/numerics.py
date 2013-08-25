#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import matplotlib
#matplotlib.use('pgf')
import numpy as np
from matplotlib.pyplot import *
from scipy.misc import comb

import os
import sys
lib_path = os.path.abspath('../../Jaynes\ Cummings\ Analytic')
sys.path.append(lib_path)
lib_path = os.path.abspath('../../Hierarchical\ Equations')
sys.path.append(lib_path)

#from jcfull import JCHierarchy
#from libbath import OscillatorBath

# Some LaTeX specific parameters
PtPerIn = 72.27

# Standard columnwidth in points
columnwidth = 402.32205


# Main plotting routines ######################################################
def ScalingBehavior(outfilename, width=columnwidth, ratio=.6):  # {{{
   """
      Plots the scaling behavior of our Hierarchichal Equations for two
      different numbers of modes.
   """
   def NumberStates(Depths, modes):
      states = np.array([], dtype=int)
      for D in np.array(Depths):
         h = [comb(i + modes - 1, modes - 1, exact=1) for i in np.arange(D+1)]
         states = np.append(states, np.sum(h))
      return states

   print('Creating ScalingBehavior plot...')
   # Setup the main plot
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(0, figsize=figsize)
   ax = subplot(111, autoscale_on=False)
   subplots_adjust(left=.2, bottom=.2, right=.9, top=.88)

   Depth_Max = 12
   Depths = np.arange(1, Depth_Max)

   # Set the ticks and tick-labels
   ax.set_xticklabels(Depths)
   ax.set_xticks(Depths)

   ## Add labels
   ax.set_xlabel(r'Depth of Hierarchy')
   ax.set_ylabel(r'Number of auxilary states')
   ax.set_yscale('log')
   ax.set_yticklabels([r'$10^{%i}$' % n for n in [0, 2, 4, 6, 8, 10]])
   ax.set_yticks([10e0, 10e2, 10e4, 10e6, 10e8, 10e10])
   ax.tick_params(axis='y', which='minor', left='off', right='off')

   colors = ['r', 'b']
   for i, N in enumerate([5, 10]):
      # Triangular Shape
      num = NumberStates(Depths, N)
      plot(np.array(Depths), np.array(num), 'x', color=colors[i],
            label='Tri., N={}'.format(N))
      # Rectangular Shape
      num = (Depths + 1.)**N
      plot(np.array(Depths), np.array(num), 'o', color=colors[i],
            label='Cub., N={}'.format(N))

   ax.axis([.5, Depth_Max - 1 + .5, 10e-1, 10e11])
   legend(ncol=2, loc='upper right')
   savefig(outfilename)
   print('Success.')
#}}}

if __name__ == '__main__':
   ScalingBehavior('scaling.pgf')
