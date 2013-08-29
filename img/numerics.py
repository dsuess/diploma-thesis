#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib.pyplot import *
from scipy.misc import comb
import cPickle as pickle

import sys
sys.path.append('/home/dsuess/Documents/Diplomarbeit/Exp Expansion/')

#from jcfull import JCHierarchy
#from libbath import OscillatorBath

# Some LaTeX specific parameters
PtPerIn = 72.27

# Standard columnwidth in points
columnwidth = 402.32205

# file format
outformat = 'pdf'


## Helper functions ###########################################################

def NumberStates(Depths, modes):  # {{{1
   """
      Calculates the number of auxilary states required given for an integer
      number of modes and and integer-array depth.
   """
   states = np.array([], dtype=int)
   for D in np.array(Depths):
      h = [comb(i + modes - 1, modes - 1, exact=1) for i in np.arange(D+1)]
      states = np.append(states, np.sum(h))
   return states


def SpinBosonSigmaZ(filename):  # {{{1
   """ Loads the dynamics of a Spin-Boson simulation from $filename.
       Returns the appropriate time and <σ_z> array as well as the options used
       in the creation.
   """
   infile = open(filename)
   options = pickle.load(infile)
   rho = pickle.load(infile)
   infile.close()
   tSteps = options['tSteps']
   tLength = options['tLength']
   t = np.linspace(0, tLength, tSteps)
   s_z = rho[:, 0, 0].real - rho[:, 1, 1].real

   return t, s_z, options

#1}}}


# Main plotting routines ######################################################

def LinVsNonlin(outfilename='linnonlin', width=columnwidth, ratio=.8):  # {{{1
   """
      Plots to compare lin vs. nonlin method.
   """
   print('Creating lin vs nonlin plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.15, right=.89, top=.99, wspace=0.,
         hspace=.03)

   axlintop = subplot(221, autoscale_on=True)
   axnontop = subplot(222, autoscale_on=True)
   axlinbot = subplot(223, autoscale_on=True)
   axnonbot = subplot(224, autoscale_on=True)

   # Plotting the data

   ## Strong coupling
   for i in [0, 1, 2]:
      t, sz, opt = SpinBosonSigmaZ('spinboson/lin{}.pkl'.format(i))
      p, = axlinbot.plot(t, sz)
      t, sz, opt = SpinBosonSigmaZ('spinboson/nonlin{}.pkl'.format(i))
      axnonbot.plot(t, sz, color=p.get_color(),
            label=r'N={}'.format(opt['Trajectories']))

   # Setup the subplot
   ## Only left ones
   for ax in [axlintop, axlinbot]:
      ax.set_ylabel(r'$\langle\sigma_z\rangle$')

   ## Only right ones
   for ax in [axnontop, axnonbot]:
      ax.set_yticklabels([])

   ## Only top ones
   for ax in [axlintop, axnontop]:
      ax.set_xticklabels([])

   ## Only bottom ones
   for ax in [axlinbot, axnonbot]:
      ax.set_xlabel(r'$t$')

   ## All of dem
   for ax in [axlintop, axnontop, axlinbot, axnonbot]:
      ax.axis([0, 50, -1.1, 1.1])

   axnonbot.legend(loc='lower right')
   savefig('.'.join([outfilename, outformat]))
   savefig('.'.join([outfilename, outformat]))

   close()
   print('Success.')
   pass


def Lorentzians(outfilename='lorentzian', width=columnwidth, ratio=.35):  # {{{1
   """
      Plots the scaling behavior of our Hierarchichal Equations for two
      different numbers of modes.
   """
   clf()
   print('Creating Lorentzians plot...')
   # Setup the main plot
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   ax = subplot(111, autoscale_on=False)
   subplots_adjust(left=.11, bottom=.15, right=.89, top=.99)

   lorentzian = lambda w, g, W: g / np.pi / ((w - W)**2 + g**2)

   w_min, w_max = -5, 11.5
   w = np.linspace(w_min, w_max, 1000)
   # Set the ticks and tick-labels
   ax.set_yticks([0, .1, .2, .3])

   ## Add labels
   ax.set_xlabel(r'$\omega$')
   ax.set_ylabel(r'$J(\omega)$')

   # Plot the ω=0 axis
   axvline(0, ls='--', color='k')

   plot(w, lorentzian(w, 1., 5.), label=r'$\gamma=1, \Omega=5')
   plot(w, lorentzian(w, 3., 5.), label=r'$\gamma=3, \Omega=5')

   axis([w_min, w_max, 0, .35])
   legend(loc='upper left')
   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')


def Expansions(outfilename='expansions', width=columnwidth, ratio=.5,  # {{{1
      quality='low'):
   """
      Plots the correlation function using different expansions or numerical
      integration for an anti-symmetric Lorentzian.
   """
   print('Creating Expansion plot...')
   # Setup the main plot
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   subplots_adjust(left=.11, bottom=.15, right=.89, top=.85, wspace=.02,
         hspace=.1)

   beta1 = 20.
   beta2 = 2.

   ax1diff = subplot(121, autoscale_on=False)
   ax2diff = subplot(122, autoscale_on=False)
   ax1 = axes([.293, .62, .19, .2])
   ax2 = axes([.685, .62, .19, .2])

   ## Do the main plotting work
   if quality == 'high':
      t = np.linspace(0, 5.2, 5000)
      divmax = 40
      rtol = .005
      toverview = np.linspace(0, 11, 1000)
   else:
      t = np.linspace(0, 5.2, 100)
      divmax = 10
      rtol = .01
      toverview = np.linspace(0, 11, 100)

   from lorentzians import ASLorentzianSpectralDensity as ASLorentzSD
   A = ASLorentzSD(2., .5, 1., rtol=rtol)

   # exact plots
   alpha_ref1 = np.real(A.alpha_exact(t, beta1, divmax=divmax))
   alpha_ref2 = np.real(A.alpha_exact(t, beta2, divmax=divmax))

   alpha_overview_1 = np.real(A.alpha_exact(toverview, beta1, divmax=divmax))
   ax1.plot(toverview, alpha_overview_1, lw=.8, color='k')
   ax1.text(6, 2, r'$\beta = {}'.format(beta1), fontsize=8)
   alpha_overview_2 = np.real(A.alpha_exact(toverview, beta2, divmax=divmax))
   ax2.plot(toverview, alpha_overview_2, lw=.8, color='k')
   ax2.text(6.5, 2, r'$\beta = {}'.format(beta2), fontsize=8)

   # pade expansion
   for T in [2, 4]:
      alpha = np.real(A.alpha_pade(t, beta1, Terms=T))
      p, = ax1diff.plot(t, np.abs(alpha_ref1 - alpha), ls='--',
            #label=', '.join([r'Pad\'{e}', '$T={}$'.format(T)]))
            label=' '.join([r'Pad\'{e}', '({})'.format(T)]))

      alpha = np.real(A.alpha_pade(t, beta2, Terms=T))
      ax2diff.plot(t, np.abs(alpha_ref2 - alpha), ls='--', color=p.get_color())

   # matsubara expansion
   for T in [2, 10]:
      alpha = np.real(A.alpha_matsubara(t, beta1, Terms=T))
      p, = ax1diff.plot(t, np.abs(alpha_ref1 - alpha), ls=':',
            label=r'Mat.~({})'.format(T))

      alpha = np.real(A.alpha_matsubara(t, beta2, Terms=T))
      ax2diff.plot(t, np.abs(alpha_ref2 - alpha), ls=':', color=p.get_color())

   ## Setup plots
   for ax in [ax1, ax2]:
      ax.set_xticks([0, 5, 10])
      ax.set_yticks([-1, 1, 3])
      setp(ax.get_xticklabels(), fontsize=8)
      setp(ax.get_yticklabels(), fontsize=8)
      ax.tick_params(axis='x', which='major', top='off', length=2)
      ax.tick_params(axis='y', which='major', right='off', length=2)
      ax.axis([0, toverview[-1], -1.5, 3.5])
      [s.set_linewidth(0.3) for s in ax.spines.itervalues()]
      #ax.set_ylabel(r'$\alpha_\mathrm{ref}(t)$', fontsize=8)

   for ax in [ax1diff, ax2diff]:
      ax.tick_params(axis='y', which='minor', left='off', right='off')
      ax.set_yscale('log')
      ax.axis([0, t[-1], 5*10e-5, 2*10e-1])

   ax2diff.set_yticklabels([])
   ax1diff.set_xlabel(r'$t$')
   ax2diff.set_xlabel(r'$t$')
   ax1diff.set_ylabel(r'$\vert \alpha_\mathrm{ref}(t) - \alpha(t)\vert$')

   ax1diff.legend(bbox_to_anchor=(-.025, 1.03, 0.4, .902), loc=3, ncol=4,
         borderaxespad=0.)
   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')


def ScaleBehavior(outfilename='scaling', width=columnwidth, ratio=.5):  # {{{1
   """
      Plots the scaling behavior of our Hierarchichal Equations for two
      different numbers of modes.
   """
   clf()
   print('Creating ScalingBehavior plot...')
   # Setup the main plot
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   figure(figsize=figsize)
   ax = subplot(111, autoscale_on=False)
   subplots_adjust(left=.11, bottom=.13, right=.89, top=.99)

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
            label='Tri., $N={}$'.format(N))
      # Rectangular Shape
      num = (Depths + 1.)**N
      plot(np.array(Depths), np.array(num), 'o', color=colors[i],
            label='Cub., $N={}$'.format(N))

   ax.axis([.5, Depth_Max - 1 + .5, 10e-1, 10e11])
   legend(ncol=2, loc='upper right')
   savefig('.'.join([outfilename, outformat]))
   close()
   print('Success.')
# 1}}}

if __name__ == '__main__':
   #ScaleBehavior(width=.9*columnwidth)
   #Lorentzians(width=.8*columnwidth)
   LinVsNonlin()
   #Expansions(quality='low')
