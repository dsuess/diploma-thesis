#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import division
from __future__ import print_function

from settings import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
sys.path.append('/home/dsuess/Documents/Diplomarbeit/Hierarchical Equations/')

from aggregats import AggregatTransferFromFile

invcm_to_ps = 5.30883746

## Helper functions ###########################################################
def get_ishitzaki_flemming(filename, nrsites=-1, convert=False, **kwargs):
   """
      Loads the data from the Ishitzaki Flemming calcluations from the space-
      seperated file given by filename.

      If convert is set we assume the time is given in ps and convert to
      cm^-1 with the conversion factor 1 / 2πc = 5.30883746.
   """
   data = np.loadtxt(filename)
   if nrsites == -1:
      t, populations = data[:, 0], data[:, 1:].transpose()
   else:
      t, populations = data[:, 0], data[:, 1:nrsites+1].transpose()

   if convert:
      t = t / invcm_to_ps

   return t, populations

# Main plotting routines ######################################################

def fmo_transfer_2term(outfilename='fmo_transfer', temp=77, # {{{1
      width=columnwidth, ratio=.35):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating FMO transfer plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.185, right=.89, top=.97, wspace=0.,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/fmo/'
   ## Setup color cycle for plotting 4 sites
   # FIXME Match colors with pymol
   cmap = ['r', 'g', 'b', 'm', 'c', '#ffa500', '.3', 'y']

   ax1 = plt.subplot(121, autoscale_on=False)
   ax2 = plt.subplot(122, autoscale_on=False)

   # Main plotting (left plot = start at 1) ###################################
   plots = []
   ## Ishi-Fleming
   t, pop = get_ishitzaki_flemming(dirloc +
         'IshFl_transfer_{}K_start_1_curve_1-4.dat'.format(temp))
   for i in range(4):
      plots.append(ax1.plot(t, pop[i], color=cmap[i], label=i+1)[0])

   ## Delete last entry (BChl 3 since it appears in both plots)
   plots.pop()

   ## hierarchy results
   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start1-2term-1.pkl'.format(temp))
   t = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps
   pop = A.get()
   for i in range(4):
      ax1.plot(t, pop[i], ls=':', color=cmap[i])

   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start1-2term-2.pkl'.format(temp))
   pop = A.get()
   for i in range(4):
      ax1.plot(t, pop[i], ls='--', color=cmap[i])

   # Main plotting (right plot = start at 6) ##################################
   ## Ishi-Fleming
   t, pop = get_ishitzaki_flemming(dirloc +
         'IshFl_transfer_{}K_start_6_curve_6-3.dat'.format(temp))
   for i in range(3, -1, -1):
      plots.append(ax2.plot(t, pop[i], color=cmap[5-i], label=6-i)[0])

   ## hierarchy results
   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start6-2term-1.pkl'.format(temp))
   t = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps
   pop = A.get()
   for i in [5, 4, 3, 2]:
      ax2.plot(t, pop[i], ls=':', color=cmap[i])

   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start6-2term-2.pkl'.format(temp))
   pop = A.get()
   for i in [5, 4, 3, 2]:
      ax2.plot(t, pop[i], ls='--', color=cmap[i])


   # Setup plot ###############################################################
   ax1.set_xlabel(r't [ps]')
   ax2.set_xlabel(r't [ps]')
   ax1.set_ylabel(r'population')

   ax1.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
   ax2.xaxis.set_major_locator(MaxNLocator(5))
   ax2.set_yticklabels([])

   #ax1.legend(ncol=2, title='BChls')
   #ax2.legend(ncol=2, title='BChls')
   ax1.legend(ncol=2)
   ax2.legend(ncol=2)

   #ax1.legend(plots, range(1, 7), bbox_to_anchor=(-.025, 1.03, 0.4, .902),
   #      loc=3, ncol=8, borderaxespad=0.)

   plt.savefig('.'.join([outfilename, outformat]))
   print('/home/dsuess/Documents/Diplomarbeit/Thesis/img/fmo_transfer.pdf')
   ax1.axis([0, 1.0, 0, 1.0])
   ax2.axis([0, 1.0, 0, 1.0])
   plt.legend()
   plt.close()
   print('Success.')

def fmo_transfer_details(outfilename='fmo_transfer', temp=77, # {{{1
      width=columnwidth, ratio=.8):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating FMO detailed transfer plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.185, right=.89, top=.97, wspace=0.,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/fmo/'
   ## Setup color cycle for plotting 4 sites
   # FIXME Match colors with pymol
   cmap = ['r', 'g', 'b', 'm', 'c', '#ffa500', '.3', 'y']

   ax1l = plt.subplot(221, autoscale_on=False)
   ax1r = plt.subplot(222, autoscale_on=False)
   ax2l = plt.subplot(223, autoscale_on=False)
   ax2r = plt.subplot(224, autoscale_on=False)

   # Main plotting (left plot = start at 1) ###################################
   ## Ishi-Fleming
   t_if, pop_if = get_ishitzaki_flemming('fmo/IshFl_transfer_{}K_start_1_curve_1-4.dat'.format(temp))
   ## Hieararchy Results
   A = AggregatTransferFromFile('fmo/fmo{}-start1-2term-1.pkl'.format(temp))
   t_1 = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps
   pop_1 = A.get()
   A = AggregatTransferFromFile('fmo/fmo{}-start1-2term-2.pkl'.format(temp))
   pop_2 = A.get()

   for i in range(4):
      ax1.plot(t_if, pop_if[i], color=cmap[i], label=i+1)
      ax1.plot(t_1, pop_1[i], ls=':', color=cmap[i])
      ax1.plot(t_1, pop_2[i], ls='--', color=cmap[i])

   # Main plotting (right plot = start at 6) ##################################
   ## Ishi-Fleming
   t_if, pop_if = get_ishitzaki_flemming('fmo/IshFl_transfer_{}K_start_1_curve_1-4.dat'.format(temp))
   ## Hieararchy Results
   A = AggregatTransferFromFile('fmo/fmo{}-start6-2term-1.pkl'.format(temp))
   t_1 = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps
   pop_1 = A.get()
   A = AggregatTransferFromFile('fmo/fmo{}-start6-2term-2.pkl'.format(temp))
   pop_2 = A.get()

   for i in range(3, -1, -1):
      ax2.plot(t_if, pop_if[i], color=cmap[5 - i], label=6 - i)
      ax2.plot(t_1, pop_1[i + 2], ls=':', color=cmap[i + 2])
      ax2.plot(t_1, pop_2[i + 2], ls='--', color=cmap[i + 2])

   # Setup plot ###############################################################
   ax1.set_xlabel(r't [ps]')
   ax2.set_xlabel(r't [ps]')
   ax1.set_ylabel(r'population')

   ax1.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
   ax2.xaxis.set_major_locator(MaxNLocator(5))
   ax2.set_yticklabels([])

   #ax1.legend(ncol=2, title='BChls')
   #ax2.legend(ncol=2, title='BChls')
   ax1.legend(ncol=2)
   ax2.legend(ncol=2)

   #ax1.legend(plots, range(1, 7), bbox_to_anchor=(-.025, 1.03, 0.4, .902),
   #      loc=3, ncol=8, borderaxespad=0.)

   plt.savefig('.'.join([outfilename, outformat]))
   print('/home/dsuess/Documents/Diplomarbeit/Thesis/img/fmo_transfer.pdf')
   ax1.axis([0, 1.0, 0, 1.0])
   ax2.axis([0, 1.0, 0, 1.0])
   plt.legend()
   plt.close()
   print('Success.')
#1}}}

if __name__ == '__main__':
   fmo_transfer_2term(temp=300, ratio=.32)
