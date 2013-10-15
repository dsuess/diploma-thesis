#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import division
from __future__ import print_function

from settings import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
import re

import sys
sys.path.append('/home/dsuess/Documents/Diplomarbeit/Hierarchical Equations/')

import functions_ger as fg
from aggregats import AggregatSpectrum, AggregatSpectrumFromFile
from libbath import OscillatorBath

dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/spectra/'

invcm_to_ps = 5.30883746

###############################################################################
def PlotSpectraComp(ax, V, wmin, wmax, sigma_w=0, Depth=3, side='left'):
   """@todo: Docstring for PlotSpectraComp.

   :V: @todo
   :X: @todo
   :gamma: @todo
   :Omega: @todo
   :returns: @todo

   """
   if side == 'right':
      pseudom = np.loadtxt(dirloc + 'pseudo{}.dat'.format(V), usecols=(1,2,3))
   else:
      pseudom = np.loadtxt(dirloc + 'out_n2m1lz1v{}dt0.01nvq1000u0_20o0_20x0._.64om0.__1.gam0.1_Fig1__mit1.dat'.format(V), usecols=(1,2,3))

   t = pseudom[:, 0]
   Corr = pseudom[:, 1] + 1.j*pseudom[:, 2]
   dt = t[1] - t[0]
   A = fg.fourier(Corr, dt, output_w=True, hermitian=True, sigma_w=sigma_w)
   bounds = (A[0] > wmin) * (A[0] < wmax)
   ax.plot(A[0][bounds] - .64, np.real(A[1][bounds]), color='r')

   if side == 'right':
      bath = OscillatorBath(.64, .25, 1.0)
   else:
      bath = OscillatorBath(.64, .1, 1.0)

   h = float(V) * np.array([[0, 1], [1, 0]])
   A = AggregatSpectrum(h, bath, Depth)

   A.calc(dw=.05, tSteps=10000)
   A.plot(wmin=wmin, wmax=wmax, ax=ax, ls=':', color='b')
   A.Depth += 2
   A.calc(dw=.05, tSteps=10000)
   A.plot(wmin=wmin, wmax=wmax, ax=ax, ls='--', color='b')
   A.Depth = 1
   A.calc(dw=.05, tSteps=10000)
   A.plot(wmin=wmin, wmax=wmax, ax=ax, ls='-', color='b')
   x1, x2, y1, y2 = ax.axis()
   ax.axis([-3, 4.5, 0, y2])
   ax.text(2., .8 * y2, 'V={}'.format(V))


# Main plotting routines ######################################################
def spectra_plot(outfilename='spectra', width=columnwidth, ratio=1.1):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating spectra plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.08, right=.96, top=.93, wspace=0.2,
         hspace=.05)

   axs = []
   ##
   for i,V in enumerate(['-1.5', '-0.41', '0.0', '0.44', '1.5']):
      ax = plt.subplot2grid((5, 2), (i, 1))
      axs.append(ax)
      PlotSpectraComp(ax, V, -2.5, 4.5, side='right')
      ax.set_xticklabels([])
      ax.xaxis.set_major_locator(MaxNLocator(6))
      ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))

   axs[4].set_xticklabels(['-3.0', '-1.5', '0.0', '1.5', '3.0', '4.5'])
   axs[4].set_xlabel(r'$\nu / \Omega$')
   axs[0].yaxis.set_major_locator(MaxNLocator(5))
   ax = axs[0].twiny()
   ax.set_xticklabels([])
   ax.set_xlabel(r'$\gamma = 0.25 \,\Omega$')

   axs = []
   ##
   for i,V in enumerate(['-1.5', '-0.41', '0.0', '0.44', '1.5']):
      ax = plt.subplot2grid((5, 2), (i, 0))
      axs.append(ax)
      PlotSpectraComp(ax, V, -2.5, 4.5)
      ax.set_xticklabels([])
      ax.xaxis.set_major_locator(MaxNLocator(6))
      ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))

   axs[4].set_xticklabels(['-3.0', '-1.5', '0.0', '1.5', '3.0', '4.5'])
   axs[4].set_xlabel(r'$\nu / \Omega$')
   axs[2].set_ylabel(r'Absorption coefficient $A(\nu)$ [arb. units]')
   axs[2].yaxis.set_label_coords(-.16, 0.5)
   axs[0].yaxis.set_major_locator(MaxNLocator(5))

   ax = axs[0].twiny()
   ax.set_xticklabels([])
   ax.set_xlabel(r'$\gamma = 0.1 \, \Omega$')

   plt.savefig('.'.join([outfilename, outformat]))
   plt.close()
   print('Success.')


def ptcda_plot(outfilename='ptcda', width=columnwidth, ratio=.5):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating spectra plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.17, right=.89, top=.93, wspace=0.2,
         hspace=.05)

   ax=plt.subplot(111)
   ax2 = ax.twinx()

   cls = ['b', 'r', 'y', 'c']
   for i, D in enumerate([1, 3, 5, 7]):
      A = AggregatSpectrumFromFile(dirloc + 'ptcda-{}.pkl'.format(D))
      A.plot(wmin=-1800, wmax=2800, ax=ax, color=cls[i], sigma_w=20,
            label='D={}'.format(D))

   A.plot(wmin=-3000, wmax=4000, ax=ax2, color='k', lw=.5, sigma_w=1)
   ax.set_xlabel(r'Wavenumber $[\mathrm{cm^{-1}}]$')
   ax.set_ylabel(r'Absorption $A(\nu)$ [arb. units]')
   ax.xaxis.set_major_locator(MaxNLocator(7))
   ax.yaxis.set_major_locator(MaxNLocator(4))
   ax2.yaxis.set_major_locator(MaxNLocator(4))
   ax2.set_yticklabels([])
   ax.set_xlim(-1400, 2800)
   ax2.set_ylim(0, .4)
   ax.set_ylim(0, .02)
   ax.legend(ncol=5)

   plt.savefig('.'.join([outfilename, outformat]))
   plt.close()
   print('Success.')



###############################################################################
if __name__ == '__main__':
   #spectra_plot()
   ptcda_plot()
