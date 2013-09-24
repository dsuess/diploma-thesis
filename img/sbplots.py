#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import division
from __future__ import print_function

from settings import *
import numpy as np
from matplotlib.pyplot import *
from matplotlib.ticker import MaxNLocator

import sys
sys.path.append('/home/dsuess/Documents/Diplomarbeit/Hierarchical Equations/')

from spinboson import SBHierarchyFromFile
from libnoise import libnoise as ln



## Helper functions ###########################################################

def plot_single_realization_norm(SB, ax, single_realizations):
   """
      TODO Docstring
   """
   psi0 = np.array([1, 0], dtype=complex)
   SB.init_single_realizations()
   for i in xrange(single_realizations):
      t, psit = SB.get_single_realization(psi0, linear=True)
      normsq = np.real(np.sum(np.conj(psit) * psit, axis=1))
      ax.plot(t, np.sqrt(normsq))
   SB.free_single_realizations()

# Main plotting routines ######################################################

def lin_norm_comparisson(outfilename='normcomp', # {{{1
      single_realizations=5, width=columnwidth, ratio=.3):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating norm comparisson plot...')
   ln.initrandomseed()
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.19, right=.89, top=.97, wspace=0.25,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/spinboson/'

   axweak = subplot(121, autoscale_on=True)
   axstrong = subplot(122, autoscale_on=True)

   # Plotting the data

   ## Weak coupling
   SB = SBHierarchyFromFile(dirloc + 'sbgan-lin-1000.pkl')
   plot_single_realization_norm(SB, axweak, single_realizations)

   SB = SBHierarchyFromFile(dirloc + 'sbstrong-lin-10000.pkl')
   plot_single_realization_norm(SB, axstrong, single_realizations)

   # Setup the subplot

   axweak.axhline(1., color='.5', ls=':', lw=1.)
   axstrong.axhline(1., color='.5', ls=':', lw=1.)
   axweak.set_ylabel(r'$\lfloor\psi_t(Z^*)\rfloor$')
   axweak.xaxis.set_major_locator(MaxNLocator(5))
   axweak.yaxis.set_major_locator(MaxNLocator(5))

   axstrong.xaxis.set_major_locator(MaxNLocator(5))
   axstrong.yaxis.set_major_locator(MaxNLocator(5))

   ## Set location of xlabel to center
   axweak.set_xlabel(r'$t\cdot\Delta$')
   axstrong.set_xlabel(r'$t\cdot\Delta$')
   #axweak.xaxis.set_label_coords(1.0, -.15)

   ## Set text on the top
   #axweaktwin = axweak.twiny()
   #axweaktwin.set_xticklabels([])
   #axweaktwin.set_xlabel('Weakly Coupled')
   #
   #axstrongtwin = axstrong.twiny()
   #axstrongtwin.set_xticklabels([])
   #axstrongtwin.set_xlabel('Strongly Coupled')

   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')


def lin_vs_nonlin_averaged(outfilename='linvsnonlin_averaged', # {{{1
      width=columnwidth, ratio=1.):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating lin vs nonlin plot (sigma_z)...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.09, right=.89, top=.94, wspace=0.,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/spinboson/'

   axlintop = subplot(221, autoscale_on=True)
   axnontop = subplot(222, autoscale_on=True)
   axlinbot = subplot(223, autoscale_on=True)
   axnonbot = subplot(224, autoscale_on=True)

   # Plotting the data

   ## Weak coupling
   for i in [10, 100, 1000]:
      Alin = SBHierarchyFromFile(dirloc + 'sbgan-lin-%i.pkl' % i)
      t, avgs = Alin.get_sigma_avg()
      axlintop.plot(t, avgs[2])

      Anon, opt = SBHierarchyFromFile(dirloc + 'sbgan-nonlin-%i.pkl' % i,
            with_opt=True)
      t, avgs = Anon.get_sigma_avg()
      axnontop.plot(t, avgs[2], label=i)
   print('Options for weak coupling are:')
   print(opt)

   for i in [100, 1000, 10000]:
      Alin = SBHierarchyFromFile(dirloc + 'sbstrong-lin-%i.pkl' % i)
      t, avgs = Alin.get_sigma_avg()
      axlinbot.plot(t, avgs[2])

      Anon, opt = SBHierarchyFromFile(dirloc + 'sbstrong-nonlin-%i.pkl' % i,
            with_opt=True)
      t, avgs = Anon.get_sigma_avg()
      axnonbot.plot(t, avgs[2], label=i)
   print('Options for strong coupling are:')
   print(opt)

   # Setup the subplot
   ## Only left ones
   for ax in [axlintop, axlinbot]:
      ax.set_ylabel(r'$\langle\sigma_z\rangle$')
      ## Remove the highest ticklabel
      ax.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))

   ## Only right ones
   for ax in [axnontop, axnonbot]:
      ax.set_yticklabels([])
      ax.xaxis.set_major_locator(MaxNLocator(5))


   ## Only the top ones
   for ax in [axlintop, axnontop]:
      ax.axis([0, 200, -1.05, 1.05])
      ax.set_xticklabels([])

   ## Only the bottom ones
   for ax in [axlinbot, axnonbot]:
      ax.axis([0, 50, -1.05, 1.05])
      ax.set_xlabel(r'$t\cdot\Delta$')

   ## Set location of xlabel to center
   #axlinbot.set_xlabel(r'$t\cdot\Delta$')
   #axlinbot.xaxis.set_label_coords(1.0, -.15)

   ## Set text on right hand side
   axnontoptwin = axnontop.twinx()
   axnontoptwin.set_yticklabels([])
   axnontoptwin.set_ylabel('Weakly Coupled')

   axnonbottwin = axnonbot.twinx()
   axnonbottwin.set_yticklabels([])
   axnonbottwin.set_ylabel('Stronly Coupled')

   ## Set text on the top
   axlintoptwin = axlintop.twiny()
   axlintoptwin.set_xticklabels([])
   axlintoptwin.set_xlabel('Linear Equations')

   axnontoptwin = axnontop.twiny()
   axnontoptwin.set_xticklabels([])
   axnontoptwin.set_xlabel('Nonlinear Equations')


   axnontop.legend(loc='upper right')
   axnonbot.legend(loc='upper right')
   savefig('.'.join([outfilename, outformat]))

   close()
   print('Success.')

#1}}}

if __name__ == '__main__':
   # TODO Adjust parameters such that times are the same!
   lin_vs_nonlin_averaged()
   lin_norm_comparisson(single_realizations=6)
