#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from settings import PtPerIn, columnwidth, outformat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from scipy.integrate import cumtrapz

import sys
sys.path.append(
      '/home/dsuess/Documents/Diplomarbeit/Jaynes Cummings Analytic/'
)

from jcfull import JCAnalytical


# Helper function
def get_F(jca, t):
   """Returns F(t) = ∫_0^t α(t-s) ψ^+(s) / ψ^-(t) ds for a given jc-calculation

   :jca: type JCAnalytical, it must have already run once
   :t: array of evenly spaced times
   :returns: F(t), same shape as t

   """
   dt = t[1] - t[0]
   integrand = jca.psi_plus[None, :] / jca.psi_plus[:, None] \
         * jca._alpha_wiggle(t[:, None] - t[None, :])
   return cumtrapz(integrand, dx=dt)[:, -1]


# Main plotting routines ######################################################
def jaynes_cummings_plot(outfilename='jaynescummings.' + outformat,
      width=columnwidth, ratio=.4):
   """
      Plot Jaynes-Cummings model: <σ>_z and F(t) where
               F(t) = ∫_0^t α(t-s) ψ^+(s) / ψ^-(t) ds
   """
   print('Creating Jaynes-Cumming plot...')
   figsize = (width / PtPerIn, width * ratio / PtPerIn)
   plt.figure(figsize=figsize)
   plt.subplots_adjust(left=.11, bottom=.19, right=.89, top=.97, wspace=0.0,
         hspace=.05)

   matplotlib.rcParams['axes.color_cycle'] = ['b']
   axdamp_s = plt.subplot(121, autoscale_on=True)
   axfree_s = plt.subplot(122, autoscale_on=True)
   axdamp_F = plt.axes([.285, .62, .2, .3], autoscale_on=True)
   axfree_F = plt.axes([.675, .62, .2, .3], autoscale_on=True)

   # Plotting the data
   ## Strongly damped
   psi0 = np.array([1, 0], dtype=complex)
   gamma = 4.
   g = gamma/2.
   omega = 1.
   coupl = np.sqrt(2)

   Omega = 1.0
   jc = JCAnalytical(omega, g, gamma, Omega, coupl)
   t, sz = jc.solve(20, 1000, psi0)
   axdamp_s.plot(t, sz)
   axdamp_F.plot(t, np.abs(jc.get_F(t)))

   Omega = 1.01
   jc = JCAnalytical(omega, g, gamma, Omega, coupl)
   t, sz = jc.solve(20, 1000, psi0)
   axdamp_s.plot(t, sz, color='r', ls='--', lw=2)
   axdamp_F.plot(t, np.abs(jc.get_F(t)), color='r', ls='--', lw=2)

   ## Weakly damped
   gamma = 0.1
   g = gamma/2.
   omega = 1.
   coupl = np.sqrt(2)

   Omega = 1.0
   jc = JCAnalytical(omega, g, gamma, Omega, coupl)
   t, sz = jc.solve(40, 1000, psi0)
   axfree_s.plot(t, sz)
   axfree_F.plot(t, np.abs(jc.get_F(t)))

   Omega = 1.01
   jc = JCAnalytical(omega, g, gamma, Omega, coupl)
   t, sz = jc.solve(40, 1000, psi0)
   axfree_s.plot(t, sz, ls='--', color='r', lw=2)
   axfree_F.plot(t, np.abs(jc.get_F(t)), ls='--', color='r', lw=2)

   # Setup the subplot
   ## reduce fontsize of inset tick-labels
   fs = 7
   for t in axdamp_F.xaxis.get_major_ticks():
      t.label.set_fontsize(fs)
   for t in axdamp_F.yaxis.get_major_ticks():
      t.label.set_fontsize(fs)
   for t in axfree_F.xaxis.get_major_ticks():
      t.label.set_fontsize(fs)
   for t in axfree_F.yaxis.get_major_ticks():
      t.label.set_fontsize(fs)

   ## for σ_z (damped)
   axdamp_s.set_xlabel(r'$\omega t$')
   axdamp_s.set_ylabel(r'$\langle\sigma_z\rangle$')
   axdamp_s.xaxis.set_major_locator(MaxNLocator(6, prune='upper'))
   axdamp_s.yaxis.set_major_locator(MaxNLocator(5))

   ## (and free)
   axfree_s.set_xlabel(r'$\omega t$')
   axfree_s.xaxis.set_major_locator(MaxNLocator(6))
   axfree_s.yaxis.set_major_locator(MaxNLocator(5))
   axfree_s.set_yticklabels([])

   ## Same for insets
   axdamp_F.set_xlabel(r'$\omega t$', fontsize=fs)
   axdamp_F.set_ylabel(r'$\vert F \vert$', fontsize=fs)
   axdamp_F.xaxis.set_minor_locator(MaxNLocator(6))
   axdamp_F.xaxis.set_major_locator(MaxNLocator(3, prune='upper'))
   axdamp_F.yaxis.set_major_locator(MaxNLocator(3))

   ## (and free)
   axfree_F.set_xlabel(r'$\omega t$', fontsize=fs)
   axfree_F.set_ylabel(r'$\vert F \vert$', fontsize=fs)
   axfree_F.xaxis.set_minor_locator(MaxNLocator(6))
   axfree_F.xaxis.set_major_locator(MaxNLocator(3, prune='upper'))
   axfree_F.yaxis.set_major_locator(MaxNLocator(4))

   ## Set plotted area
   axdamp_s.axis([0, 10, -1, 1])
   axdamp_F.axis([0, 10, 0, 1.5])


   axfree_s.axis([0, 20, -1, 1])
   axfree_F.axis([0, 20, -1, 40])

   ## And add subfigure caption
   axdamp_s.text(9, -.88, r'\textbf A', fontsize=11)
   axfree_s.text(27, -.88, r'\textbf B', fontsize=11)

   plt.savefig(outfilename)
   plt.close()
   print('Success.')


if __name__ == '__main__':
   jaynes_cummings_plot()
