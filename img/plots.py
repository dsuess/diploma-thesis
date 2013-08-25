#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('pgf')
import numpy as np
from matplotlib.pyplot import *

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
def JCPlot(outfilename='jc.pdf', width=columnwidth, ratio=.4):  # {{{
   print('Creating')
   # Setup the main plot
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(0, figsize=figsize)
   s = subplot(111, autoscale_on=True)
   subplots_adjust(left=.15, bottom=.3, right=.98, top=.97, wspace=.02,
                   hspace=.22)

   t = np.linspace(0, 10, 1000)
   plot(t, t**2, label=r'$t^2$')
   s.set_ylabel(r'Testing')
   s.set_xlabel(r'Figure 1.1.: This is a test')
   legend()
   savefig(outfilename)
   print('Done!')


   ## Set the t-axis ticks
   #ax77.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', ''])
   #ax77.set_xticks([.0, .2, .4, .6, .8, 1.])
   #ax300.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
   #ax300.set_xticks([.0, .2, .4, .6, .8, 1.])
   #
   ## And the σ_z ticks
   #ax77.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
   #ax77.set_yticks([.0, .2, .4, .6, .8, 1.])
   #ax300.set_yticklabels([])
   #ax300.set_yticks([.0, .2, .4, .6, .8, 1.])
   #
   ## Add labels
   #ax77.set_xlabel(r'$t$')
   #ax300.set_xlabel(r'$t$')
   #ax77.set_ylabel(r'$\vert \psi_n(t) \vert^2$')
   #
   #t, psi, options = OccupationProbability('fmo/fmo-77-1.pkl')
   #for i in range(4):
   #   ax77.plot(t, psi[i], label=i+1)
   #
   #ax77.axis([0, 1, 0, 1])
   #ax77.legend(ncol=2, loc='upper right')
   #print('Saving to file ' + outfilename)
   #savefig(outfilename)
   #print('Success.')
#}}}

if __name__ == '__main__':
   JCPlot('fmo.pgf')
