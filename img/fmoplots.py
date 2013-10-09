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

def get_phi(filename, sites, **kwargs):
   """
      Loads data from the Parallel Hierarchical Integrator data format
   """
   pair = re.compile(r'\(([^,\)]+),([^,\)]+)\)')

   def parse_pair(s):
      return complex(*map(float, pair.match(s.strip()).groups()))

   conv = {0: lambda s: float(s.strip())}
   for i in range(7**2):
      conv[i + 1] = parse_pair

   data = np.genfromtxt(filename, delimiter=' ', dtype=np.complex,
         converters=conv)

   t = np.array([np.real(a[0]) for a in data])
   rho = np.array([np.reshape(list(a)[1:], (sites, sites)) for a in data])

   return t, np.transpose(np.diagonal(rho, axis1=1, axis2=2))


def max_diff(pop, popdiff):
   """@todo: Docstring for get_max_diff.

   :pop: @todo
   :popdiff: @todo
   :returns: @todo

   """
   A =  np.max(np.abs(pop - popdiff), axis=0)
   return A


# Main plotting routines ######################################################

def fmo_transfer_bcf(outfilename='fmo_bcf', temp=77, # {{{1
      width=columnwidth, ratio=.25):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating FMO bath correlation function plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.26, right=.89, top=.92, wspace=0.3,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/fmo/'

   ax1 = plt.subplot(121, autoscale_on=True)
   ax2 = plt.subplot(122, autoscale_on=True)

   l = 35
   g = invcm_to_ps / (50 * 1e-3)
   print(g)
   # Plot data ################################################################
   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start1-2term-1.pkl'.format(temp))
   bath = A._bath

   ## plot spectral density
   J  = lambda w: 2 * g * l * w / (w**2 + g**2) / np.pi
   w = np.linspace(0, 4000, 10000)
   ax1.plot(w, np.real(J(w)))

   ## plot bath correlation function
   t = np.linspace(0, .2, 1000)
   a = bath._alpha(t) / 1e3
   ax2.plot(invcm_to_ps * t, np.real(a), color='r', label=r'$\Re\alpha$')
   ax2.plot(invcm_to_ps * t, np.imag(a), color='g', label=r'$\Im\alpha$')

   bath.g[1] -= 1.j * np.imag(bath.g[0])
   a = bath._alpha(t) / 1e3
   ax2.plot(invcm_to_ps * t, np.real(a), color='r', ls='--')
   ax2.plot(invcm_to_ps * t, np.imag(a), color='g', ls='--')

   # Setup plot ###############################################################

   ax1.set_xlabel(r'$\omega\,\mathrm{[cm^{-1}]}$')
   ax1.set_ylabel(r'$J(\omega)\,\mathrm{[cm^{-1}]}$')
   ax2.set_xlabel(r'$t\,\mathrm{[ps]}$')
   ax2.set_ylabel(r'$\alpha(t)\,\mathrm{[cm^{-2}]}$')


   ax1.xaxis.set_major_locator(MaxNLocator(4))
   ax1.yaxis.set_major_locator(MaxNLocator(4))
   ax2.xaxis.set_major_locator(MaxNLocator(4))
   ax2.yaxis.set_major_locator(MaxNLocator(4))
   ax1.axis([0, 2000, 0, 15])
   ax2.axis([0, .2, -4, 8])


   ax1.text(40, 12.6, r'\textbf A')
   ax2.text(.005, 6, r'\textbf B')

   ax2.legend()
   plt.savefig('.'.join([outfilename, outformat]))
   plt.close()
   print('Success.')


def fmo_mevsphi_plot(outfilename='fmo_mevsphi', width=columnwidth, ratio=.7):# {{{1
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating FMO mevsphi plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.1, right=.89, top=.93, wspace=0.,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/mevsphi/'

   #ax1 = plt.subplot(121, autoscale_on=False)
   #ax2 = plt.subplot(122, autoscale_on=False)
   #axdet1 = plt.axes([.275, .68, .2, .24], autoscale_on=False)
   #axdet2 = plt.axes([.665, .68, .2, .24], autoscale_on=False)

   gs = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[4, 4])

   ax1 = plt.subplot(gs[0], autoscale_on=False)
   ax2 = plt.subplot(gs[1], autoscale_on=False)
   axdet1 = plt.subplot(gs[2], autoscale_on=False)
   axdet2 = plt.subplot(gs[3], autoscale_on=False)

   # FIXME Match colors with pymol
   cmap = ['r', 'g', 'b', 'm', 'c', '#ffa500', '.3', 'y']
   # FIXME Different colors!
   diff_cmap = ['g', 'b', 'c', 'm', 'r']
   lss = [':', '--', '-.']
   sites = [0, 1, 2, 3]

   # Main plotting (left plot = Hierarchy) ####################################
   ## Ishi-Fleming
   t, pop = get_ishitzaki_flemming(dirloc +
         'IshFl_transfer_300K_start_1_curve_1-4.dat')
   for i in range(4):
      ax1.plot(t, pop[i], color=cmap[i])
   ax1.plot([], ls='-', color='k', label='IF')

   ## FIXME MAke this 3
   A = AggregatTransferFromFile(dirloc + 'fmo300-start1-2.pkl')
   pop_ref = A.get()
   t = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps

   ## FIXME Add third order
   for i, D in enumerate([1, 2]):
   #for i, D in enumerate([]):
      A = AggregatTransferFromFile(dirloc + 'fmo300-start1-{}.pkl'.format(D))
      pop = A.get()
      for j in sites:
         ax1.plot(t, pop[j], color=cmap[j], ls=lss[i])
      ## for the legend
      ax1.plot([], ls=lss[i], color='k', label=D)

   ## Make this [1, 2]
   for i, D in enumerate([1, 1]):
   #for i, D in enumerate([]):
      A = AggregatTransferFromFile(dirloc + 'fmo300-start1-{}.pkl'.format(D))
      pop = A.get()
      diff = max_diff(pop, pop_ref)
      axdet1.plot(t, diff, color=diff_cmap[i], label=D)
      axdet1.axhline(np.max(diff), color=diff_cmap[i], ls=':')

   A = AggregatTransferFromFile(dirloc + 'fmo300-start1-3-super.pkl')
   pop_ref = A.get()
   t = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps
   crop = t <= 1.0
   for i, D in enumerate([1, 2]):
   #for i, D in enumerate([]):
      A = AggregatTransferFromFile(dirloc + 'fmo300-start1-{}-super.pkl'.format(D))
      pop = A.get()
      diff = max_diff(pop, pop_ref)
      axdet1.plot(t, diff, color=diff_cmap[i+2], label='{}*'.format(D))
      axdet1.axhline(np.max(diff[crop]), color=diff_cmap[i+2], ls=':')

   # Main plotting (right plot = PHI) #########################################
   ## Ishi-Fleming
   t, pop = get_ishitzaki_flemming(dirloc +
         'IshFl_transfer_300K_start_1_curve_1-4.dat')
   for i in range(4):
      ax2.plot(t, pop[i], color=cmap[i])
   ## for the legend
   ax2.plot([], ls='-', color='k', label='IF')

   for i, D in enumerate([1, 2, 5]):
   #for i, D in enumerate([]):
      t, pop = get_phi(dirloc + 'fmophi300-start1-{}.dat'.format(D), 7)
      for j in sites:
         ax2.plot(t, pop[j], color=cmap[j], ls=lss[i])
      ## for the legend
      ax2.plot([], ls=lss[i], color='k', label=D)

   pop_ref = get_phi(dirloc + 'fmophi300-start1-5.dat', 7)[1]
   for i, D in enumerate([1, 2, 3, 4]):
   #for i, D in enumerate([]):
      t, pop = get_phi(dirloc + 'fmophi300-start1-{}.dat'.format(D), 7)
      diff = max_diff(pop, pop_ref)
      axdet2.plot(t, diff, color=diff_cmap[i], label=D)
      axdet2.axhline(np.max(diff), color=diff_cmap[i], ls=':')

   # Setup plot ###############################################################
   axdet1.set_yscale('log')
   axdet2.set_yscale('log')

   axdet1.set_xlabel(r't [ps]')
   axdet2.set_xlabel(r't [ps]')
   ax1.set_ylabel(r'population')
   axdet1.set_ylabel(r'deviation')

   ## left side
   for ax in [ax1, axdet1]:
      ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

   ## right side
   for ax in [ax2, axdet2]:
      ax.xaxis.set_major_locator(MaxNLocator(5))
      ax.set_yticklabels([])

   ## top
   for ax in [ax1, ax2]:
      ax.xaxis.set_major_locator(MaxNLocator(5))
      ax.set_xticklabels([])

   ## bottom
   for ax in [axdet1, axdet2]:
      ax.yaxis.set_minor_locator(LogLocator(numticks=100))

   #axdet2.set_yticklabels([])
   ax2.xaxis.set_major_locator(MaxNLocator(5))

   #for axdet in [axdet1, axdet2]:
   #   pass
   #   axdet.xaxis.set_major_locator(MaxNLocator(2))
   #   axdet.xaxis.set_minor_locator(MaxNLocator(4))
   #   axdet.yaxis.set_major_locator(MaxNLocator(2))
   #   axdet.yaxis.set_minor_locator(MaxNLocator(4))

   #   ## reduce fontsize of inset tick-labels
   #   for t in axdet.xaxis.get_major_ticks():
   #      t.label.set_fontsize(7)
   #   for t in axdet.yaxis.get_major_ticks():
   #      t.label.set_fontsize(7)

   #ax2.set_yticklabels([])
   #axdet1.set_xticklabels([])

   ## Labels on top
   ax1top = ax1.twiny()
   ax1top.set_xticklabels([])
   ax1top.set_xlabel(r'Stochastic Hierarchy')
   ax2top = ax2.twiny()
   ax2top.set_xticklabels([])
   ax2top.set_xlabel(r'\textsc{HEOM}')


   #ax1.legend(ncol=2, title='BChls')
   #ax2.legend(ncol=2, title='BChls')
   ax1.legend(ncol=2)
   ax2.legend(ncol=2)
   axdet1.legend(ncol=2, loc='upper right')
   axdet2.legend(ncol=2, loc='upper right')

   #ax1.legend(plots, range(1, 7), ncol=2)
   #ax1.legend(plots, range(1, 7), bbox_to_anchor=(-.025, 1.03, 0.4, .902),
   #      loc=3, ncol=8, borderaxespad=0.)
   #
   ax1.text(.04, .9, r'\textbf A')
   ax2.text(.04, .9, r'\textbf B')
   axdet1.text(.04, 7, r'\textbf C')
   axdet2.text(.04, 7, r'\textbf D')

   ax1.axis([0, 1.0, 0, 1.0])
   ax2.axis([0, 1.0, 0, 1.0])
   axdet1.axis([0, 1.0, 8e-4, 2e0])
   axdet2.axis([0, 1.0, 8e-4, 2e0])
   plt.savefig('.'.join([outfilename, outformat]))
   plt.close()
   print('Success.')


###############################################################################
def fmo_transfer_ishfl(outfilename='fmo_ishfl', temp=77, # {{{1
      width=columnwidth, ratio=.5):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating FMO transfer plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.17, right=.89, top=.97, wspace=0.,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/fmo/'
   ## Setup color cycle for plotting 4 sites
   # FIXME Match colors with pymol
   cmap = ['r', 'g', 'b', 'm', 'c', '#ffa500', '.3', 'y']

   ax1 = plt.subplot(121, autoscale_on=False)
   ax2 = plt.subplot(122, autoscale_on=False)
   #axdet1 = plt.axes([.3, .68, .2, .2], autoscale_on=False)
   axdet2 = plt.axes([.665, .68, .2, .24], autoscale_on=False)


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
   # And plot the "free" FMO ##################################################

   #A = AggregatTransferFromFile(dirloc + 'fmofree-start1.pkl')
   #pop = A.get()
   #for i in range(4):
      #axdet1.plot(t, pop[i], color=cmap[i])

   # Main plotting (right plot = start at 6) ##################################
   ## Ishi-Fleming
   t, pop = get_ishitzaki_flemming(dirloc +
         'IshFl_transfer_{}K_start_6_curve_6-3.dat'.format(temp))
   for i in range(3, -1, -1):
      plots.append(ax2.plot(t, pop[i], color=cmap[5-i], label=6-i)[0])

   ## hierarchy results
   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start6-1.pkl'.format(temp))
   t = np.linspace(0, A._tLength, A._tSteps) * invcm_to_ps
   pop = A.get()
   for i in [5, 4, 3, 2]:
      ax2.plot(t, pop[i], ls=':', color=cmap[i])

   A = AggregatTransferFromFile(dirloc +
         'fmo{}-start6-2.pkl'.format(temp))
   pop = A.get()
   for i in [5, 4, 3, 2]:
      ax2.plot(t, pop[i], ls='--', color=cmap[i])


   # And plot the "free" FMO ##################################################
   A = AggregatTransferFromFile(dirloc + 'fmofree-start6.pkl')
   pop = A.get()
   for i in [5, 4, 3, 2]:
      axdet2.plot(t, pop[i], color=cmap[i])


   # Setup plot ###############################################################
   ax1.set_xlabel(r't [ps]')
   ax2.set_xlabel(r't [ps]')
   ax1.set_ylabel(r'population')

   ax1.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
   ax2.xaxis.set_major_locator(MaxNLocator(5))

   #for axdet in [axdet1, axdet2]:
   for axdet in [axdet2]:
      pass
      axdet.xaxis.set_major_locator(MaxNLocator(2))
      axdet.xaxis.set_minor_locator(MaxNLocator(4))
      axdet.yaxis.set_major_locator(MaxNLocator(2))
      axdet.yaxis.set_minor_locator(MaxNLocator(4))

      ## reduce fontsize of inset tick-labels
      for t in axdet.xaxis.get_major_ticks():
         t.label.set_fontsize(7)
      for t in axdet.yaxis.get_major_ticks():
         t.label.set_fontsize(7)

   ax2.set_yticklabels([])
   #axdet1.set_xticklabels([])


   #ax1.legend(ncol=2, title='BChls')
   #ax2.legend(ncol=2, title='BChls')
   #ax1.legend(ncol=2)
   #ax2.legend(ncol=2)

   #ax1.legend(plots, range(1, 7), ncol=2)
   #ax1.legend(plots, range(1, 7), bbox_to_anchor=(-.025, 1.03, 0.4, .902),
   #      loc=3, ncol=8, borderaxespad=0.)
   #
   ax1.text(.04, .9, r'\textbf A')
   ax2.text(.04, .9, r'\textbf B')

   ax1.axis([0, 1.0, 0, 1.0])
   ax2.axis([0, 1.0, 0, 1.0])
   axdet1.axis([0, 4.0, 0, 1.0])
   axdet2.axis([0, 1.0, 0, 1.0])
   plt.savefig('.'.join([outfilename, outformat]))
   plt.close()
   print('Success.')

#1}}}

if __name__ == '__main__':
   #fmo_transfer_bcf()
   #fmo_transfer_ishfl(temp=77, ratio=.4)
   fmo_mevsphi_plot()
