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
import spinbosonNN as sbNN
import spinboson as sb
from libbath import OscillatorBath



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

def depth_plot(outfilename='depth', width=columnwidth, ratio=.55):# {{{1
   """
      Plots to compare different depths for strongly coupled system
   """
   print('Creating depth plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.13, right=.89, top=.96, wspace=0.0,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/spinboson-cutoff/'

   ax1 = subplot(221)
   ax2 = subplot(222)
   ax3 = subplot(223)
   ax4 = subplot(224)

   matplotlib.rcParams['legend.handlelength'] = 1
   psi0 = np.array([1., 0.])

   g = 0.5
   gamma = 1.0
   bath = OscillatorBath(2 * g, gamma, 0)
   SB = sb.SBHierarchy(0., 1., 1, bath, 8)
   SB.init_single_realizations(tLength=50, tSteps=10000)
   t, psit = SB.get_single_hierarchy(psi0)

   for i in [0, 1, 4, 8]:
      ax3.plot(t, np.sqrt(np.abs(psit[:, i, 0])**2 + np.abs(psit[:, i, 1])**2),
            label='k={}'.format(i))

   sz_ref = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
   SB.free_single_realizations()
   for D in [2, 4, 6]:
      SB = sb.SBHierarchy(0., 1., 1, bath, D)
      SB.init_single_realizations(tLength=50, tSteps=10000)
      t, psit = SB.get_single_hierarchy(psi0)
      sz = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
      ax1.plot(t, np.abs(sz - sz_ref), label='{}'.format(D))
      SB.free_single_realizations()
      print('Doing {}'.format(D))

   bath = OscillatorBath(g, gamma / 2, 0)
   SB = sb.SBHierarchy(0., 1., 1, bath, 8)
   SB.init_single_realizations(tLength=50, tSteps=10000)
   t, psit = SB.get_single_hierarchy(psi0)

   for i in [0, 1, 4, 8]:
      ax4.plot(t, np.sqrt(np.abs(psit[:, i, 0])**2 + np.abs(psit[:, i, 1])**2))

   sz_ref = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
   SB.free_single_realizations()
   for D in [2, 4, 6]:
      SB = sb.SBHierarchy(0., 1., 1, bath, D)
      SB.init_single_realizations(tLength=50, tSteps=10000)
      t, psit = SB.get_single_hierarchy(psi0)
      sz = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
      ax2.plot(t, np.abs(sz - sz_ref), label='{}'.format(D))
      SB.free_single_realizations()
      print('Doing {}'.format(D))

   bath = OscillatorBath(g, gamma, 0)
   SB = sb.SBHierarchy(0., 1., 1, bath, 8)
   SB.init_single_realizations(tLength=50, tSteps=10000)
   t, psit = SB.get_single_hierarchy(psi0)
   for i in [0, 1, 4, 8]:
      ax3.plot(t, np.sqrt(np.abs(psit[:, i, 0])**2 + np.abs(psit[:, i, 1])**2),
            ls=':')
      ax4.plot(t, np.sqrt(np.abs(psit[:, i, 0])**2 + np.abs(psit[:, i, 1])**2),
            ls=':')
   SB.free_single_realizations()

   ## Plot that data! #########################################################

   # Setup the subplot
   ## Only left ones
   ax1.set_xticklabels([])
   ax2.set_xticklabels([])
   ax3.set_xlabel(r'$t \cdot \Delta$')
   ax4.set_xlabel(r'$t \cdot \Delta$')
   ax1.set_ylabel(r'$\vert \langle\sigma_z\rangle - \langle\sigma_z\rangle_\mathrm{ref} \vert$')
   ax3.set_ylabel(r'$\vert \psi^{(k)}_t \vert \, / \, \vert \psi^{(0)}_t \vert$')

   #ax1.legend(ncol=3, loc='upper right')
   ax1.set_yscale('log')
   ax1.axis([0, 50, 3e-3, 1e1])
   ax2.set_yscale('log')
   ax2.axis([0, 50, 3e-3, 1e1])
   ax2.set_yticklabels([])
   ax3.set_yscale('log')
   ax3.axis([0, 50, 5e-4, 3])
   ax4.set_yscale('log')
   ax4.axis([0, 50, 5e-4, 3])
   ax4.set_yticklabels([])


   ax1.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
   ax2.xaxis.set_major_locator(MaxNLocator(5))
   ax1.yaxis.set_major_locator(LogLocator(numticks=5))
   ax2.yaxis.set_major_locator(LogLocator(numticks=5))
   ax1.yaxis.set_minor_locator(LogLocator(numticks=5))
   ax2.yaxis.set_minor_locator(LogLocator(numticks=5))
   ax3.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
   ax4.xaxis.set_major_locator(MaxNLocator(5))
   ax3.yaxis.set_major_locator(LogLocator(numticks=5))
   ax4.yaxis.set_major_locator(LogLocator(numticks=5))
   ax3.yaxis.set_minor_locator(LogLocator(numticks=5))
   ax4.yaxis.set_minor_locator(LogLocator(numticks=5))

   ax1.text(2, 3.3, r'\textbf{A}')
   ax2.text(2, 3.3, r'\textbf{B}')
   ax3.text(2, 1.1, r'\textbf{C}')
   ax4.text(2, 1.1, r'\textbf{D}')
   ax1.legend(ncol=3)
   ax3.legend(ncol=2, loc='lower right')

   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')



def depth_plot_terminator(outfilename='terminator', width=columnwidth, ratio=.3):# {{{1
   """
      Plots to compare different depths for strongly coupled system
   """
   print('Creating depth plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.19, right=.89, top=.94, wspace=0.0,
         hspace=.05)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/spinboson-cutoff/'

   ax1 = subplot(121)
   ax2 = subplot(122)

   matplotlib.rcParams['legend.handlelength'] = 1
   psi0 = np.array([1., 0.])

   #g, gamma, Omega = sb.GANToOurs(0.004, 1., 0.0154)
   #bath = OscillatorBath(2., .5, 2.)
   g = 0.5
   gamma = 1.
   Omega = 0
   bath = OscillatorBath(g, gamma, Omega)
   SB = sb.SBHierarchy(0., 1., 1, bath, 8)
   SB.init_single_realizations(tLength=50, tSteps=10000)
   t, psit = SB.get_single_hierarchy(psi0)
   sz_ref = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
   SB.free_single_realizations()
   for D in [2, 4, 6]:
      SB = sb.SBHierarchy(0., 1., 1, bath, D)
      SB.init_single_realizations(tLength=50, tSteps=10000)
      t, psit = SB.get_single_hierarchy(psi0)
      sz = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
      ax1.plot(t, np.abs(sz - sz_ref), label='{}'.format(D))
      SB.free_single_realizations()
      print('Doing {}'.format(D))

   SBNN = sbNN.SBHierarchy(0., 1., 1, bath, 8)
   SBNN.init_single_realizations(tLength=50, tSteps=5000)
   t, psit = SBNN.get_single_hierarchy(psi0)
   sz_ref = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
   SBNN.free_single_realizations()
   for D in [2, 4, 6]:
      SBNN = sbNN.SBHierarchy(0., 1., 1, bath, D)
      SBNN.init_single_realizations(tLength=50, tSteps=5000)
      t, psit = SBNN.get_single_hierarchy(psi0)
      sz = np.abs(psit[:, 0, 0])**2 - np.abs(psit[:, 0, 1])**2
      ax2.plot(t, np.abs(sz - sz_ref), label='{}'.format(D))
      SBNN.free_single_realizations()
      print('Doing {}'.format(D))

   ## Plot that data! #########################################################

   # Setup the subplot
   ## Only left ones
   ax1.set_xlabel(r'$t \cdot \Delta$')
   ax2.set_xlabel(r'$t \cdot \Delta$')
   ax1.set_ylabel(r'$\vert \langle\sigma_z\rangle - \langle\sigma_z\rangle_\mathrm{ref} \vert$')

   #ax1.legend(ncol=3, loc='upper right')
   ax1.set_yscale('log')
   ax1.axis([0, 50, 1e-5, 10])
   ax2.set_yscale('log')
   ax2.axis([0, 50, 1e-5, 10])
   ax2.set_yticklabels([])


   ax1.xaxis.set_major_locator(MaxNLocator(5, prune='upper'))
   ax2.xaxis.set_major_locator(MaxNLocator(5))
   ax1.yaxis.set_major_locator(LogLocator(numticks=5))
   ax2.yaxis.set_major_locator(LogLocator(numticks=5))
   ax1.yaxis.set_minor_locator(LogLocator(numticks=5))
   ax2.yaxis.set_minor_locator(LogLocator(numticks=5))

   ax1.text(2, 2.0, r'\textbf{A}')
   ax2.text(2, 2.0, r'\textbf{B}')
   ax1.legend(ncol=3)

   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')


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
   matplotlib.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'm', 'c', '#ffa500', '.3', 'y']


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

   x1, x2, y1, y2 = axweak.axis()
   axweak.axis([0, 200, 0, y2])
   axweak.text(180, .9 * y2, r'\textbf{A}')

   x1, x2, y1, y2 = axstrong.axis()
   axstrong.axis([0, 50, 0, y2])
   axstrong.text(45, .9 * y2, r'\textbf{B}')

   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')


def lin_vs_nonlin_averaged(outfilename='linvsnonlin_averaged', # {{{1
      width=columnwidth, ratio=.75):
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating lin vs nonlin plot (sigma_z)...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.09, right=.89, top=.94, wspace=0.,
         hspace=.15)
   dirloc = '/home/dsuess/Documents/Diplomarbeit/archive/spinboson/'

   matplotlib.rcParams['axes.color_cycle'] = ['y', 'b', 'r']

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
      #ax.set_xticklabels([])

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
   axlintoptwin.set_xlabel('Linear Equation')

   axnontoptwin = axnontop.twiny()
   axnontoptwin.set_xticklabels([])
   axnontoptwin.set_xlabel('Nonlinear Equation')


   axnontop.legend(loc='upper right')
   axnonbot.legend(loc='upper right')
   axlintop.text(180, -.95, r'\textbf{A}')
   axnontop.text(180, -.95, r'\textbf{B}')
   axlinbot.text(45, -.95, r'\textbf{C}')
   axnonbot.text(45, -.95, r'\textbf{D}')

   savefig('.'.join([outfilename, outformat]))

   close()
   print('Success.')

#1}}}

if __name__ == '__main__':
   lin_vs_nonlin_averaged()
   #lin_norm_comparisson(single_realizations=6)
   #depth_plot()
   #depth_plot_terminator()
