import math
import copy
import time
import numba
import numpy as np
import scipy.ndimage
import numpy.random
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
from io import StringIO

#mpl.style.use("classic")
#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#mpl.rcParams.update({'font.size': 22})
plt.rc('font', size = 28,family="sans-serif")
#plt.rc('text', usetex = True)
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

fig = plt.figure(figsize = (14, 10))
plt.subplots_adjust(hspace=0.6,wspace=0.5)
gs = gridspec.GridSpec(3, 2)
ax00=fig.add_subplot(gs[0:2,0:2])
ax10=fig.add_subplot(gs[-1,0])
ax11=fig.add_subplot(gs[-1,1])

kxx=1.

def T1xy(kyy,kxy,kyx,r):
    return (kyx**2*(kyx**2*r**2-2*kxy*kyx*r+(kxx+kyy)**2*r+kxy**2))/4/(kxx+kyy)/(kyx**2*r-kxy*kyx+kxx*(kxx+kyy))
def Txy(kyy,kxy,kyx,r):
    return 1/2*((kxx**2+r*kyx**2)**0.5-kxx)

x00=np.loadtxt('data_experiment_t12.txt',skiprows=1)
x01=np.loadtxt('data_experiment_t21.txt',skiprows=1)

yg=np.loadtxt('kdependent_gaussian_data.txt',skiprows=1)
yt=np.loadtxt('kdependent_tepws_data.txt',skiprows=1)

ax00.plot(yt[:,0],yt[:,1],marker='None',color='k',linestyle='-',lw=2,fillstyle='none',label=r'$\mathrm{TE-PWS}$')
ax00.fill_between(yt[:,0],yt[:,1]-2*yt[:,2],yt[:,1]+2*yt[:,2],color='k',alpha=0.2)
ax00.plot(yg[:,0],yg[:,1],color='r',lw=2.5,label=r'$\mathrm{Gaussian}$')
ax00.fill_between(yg[:,0],yg[:,1]-2*yg[:,2],yg[:,1]+2*yg[:,2],color='r',alpha=0.4)
ax00.hlines(T1xy(1,0.9,0.9,0.2),yg[0,0],yg[-1,0],color='r',linestyle='-.',lw=2.5)
ax00.hlines(Txy(1,0.9,0.9,0.2),yg[0,0],yg[-1,0],color='r',linestyle='--',lw=2.5)

unit=0.23
ax10.errorbar(x00[:,0],x00[:,1],yerr=x00[:,2],marker='o',color='b',ms=12,fillstyle='none',mew=2,linestyle='None')
ax10.hlines(0.053,x00[0,0],x00[-1,0],color='k',linestyle='--',lw=3)
#plt.xscale('log')
ax10.set_xlim(0,x00[-1,0])
ax10.set_ylim(-0.005,0.125)
ax10.set_xlabel(r'$T~[\mu_{0}^{-1}]$')
ax10.set_ylabel(r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}$'+'\n'+'$[\mathrm{nats}~\mu_{0}]$')
#plt.tight_layout()
#plt.show()
#np.savetxt('data_experiment_t21.txt',np.stack((ts[::s]*mu0unit,Tn[::s,0]/ts[::s]/mu0unit,Tne[::s,0]/ts[::s]/mu0unit),axis=-1))

ax11.errorbar(x01[:,0],x01[:,1],yerr=x01[:,2],marker='s',color='g',ms=12,fillstyle='none',mew=2,linestyle='None')
ax11.hlines(0.008,x01[0,0],x01[-1,0],color='k',linestyle='--',lw=3)
#plt.xscale('log')
ax11.set_xlim(0,x01[-1,0])
ax11.set_ylim(-0.005,0.055)
ax11.set_xlabel(r'$T~[\mu_{0}^{-1}]$')
ax11.set_ylabel(r'$\dot{\mathcal{T}}_{X_{2}\to X_{1}}$'+'\n'+'$[\mathrm{nats}~\mu_{0}]$')
ax00.set_xscale('log')
ax00.set_xlim(yg[0,0],yg[-1,0])
ax00.set_xticks([0.01,1])
ax00.set_xlabel(r'$(k+1)\delta t/\tau$',labelpad=-20)
ax00.set_ylabel(r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}^{~[k]}~[\mathrm{nats~}a_{11}]$')

ax00.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.02, 0.62),handletextpad=0.2)

ax00.annotate(r'$\mathrm{Model~A}$',xy=(0.83,0.9),xycoords='axes fraction')
ax10.annotate(r'$\mathrm{Model~B}$',xy=(0.6,0.8),xycoords='axes fraction')
ax11.annotate(r'$\mathrm{Model~B}$',xy=(0.6,0.8),xycoords='axes fraction')

#plt.show()
ax10.annotate(r'$(a)$',xy=(-0.42,4.),xycoords='axes fraction')
#ax01.annotate(r'$(b)$',xy=(-0.15,1.1),xycoords='axes fraction')
ax10.annotate(r'$(b)$',xy=(-0.42,1.04),xycoords='axes fraction')
ax11.annotate(r'$(c)$',xy=(-0.49,1.04),xycoords='axes fraction')
plt.savefig('fig_validation.png',bbox_inches='tight',pad_inches=0.1)

