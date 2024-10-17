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
plt.rc('font', size = 30)
plt.rc('text', usetex = True)

fig = plt.figure(figsize = (14, 7.5))
plt.subplots_adjust(hspace=0.65,wspace=0.35)
gs = gridspec.GridSpec(2, 2)
ax00=fig.add_subplot(gs[0,0])
ax01=fig.add_subplot(gs[0,1])
ax10=fig.add_subplot(gs[1,0])
ax11=fig.add_subplot(gs[1,1])

kxx=1.
kzz=1.5
kyy=2.
kxz=0.0
kzx=0.2
kzy=0.3
kyz=0.0
kyx=0.5
kxy=0.5
Dx=1.5
Dz=1.
Dy=0.5

def Tyx():
    return 1/2*((Dy*kxy**2/Dx+kyy**2)**0.5-kyy)
def Txy():
    return 1/2*((Dx*kyx**2/Dy+kxx**2)**0.5-kxx)

x00=np.loadtxt('data_experiment_t12.txt',skiprows=1)
x01=np.loadtxt('data_experiment_t21.txt',skiprows=1)
x10=np.loadtxt('data_analytical_t12.txt',skiprows=1)
x11=np.loadtxt('data_analytical_t21.txt',skiprows=1)

unit=0.23
ax00.errorbar(x00[:,0],x00[:,1],yerr=x00[:,2],marker='o',color='b',ms=12,fillstyle='none',mew=2,linestyle='None')
ax00.hlines(0.053,x00[0,0],x00[-1,0],color='k',linestyle='--',lw=3)
#plt.xscale('log')
ax00.set_xlim(0,x00[-1,0])
ax00.set_ylim(-0.005,0.125)
ax00.set_xlabel(r'$T~[\mu_{0}^{-1}]$')
ax00.set_ylabel(r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}~[\mathrm{nats}~\mu_{0}]$')
#plt.tight_layout()
#plt.show()
#np.savetxt('data_experiment_t21.txt',np.stack((ts[::s]*mu0unit,Tn[::s,0]/ts[::s]/mu0unit,Tne[::s,0]/ts[::s]/mu0unit),axis=-1))

ax01.errorbar(x01[:,0],x01[:,1],yerr=x01[:,2],marker='s',color='r',ms=12,fillstyle='none',mew=2,linestyle='None')
ax01.hlines(0.008,x01[0,0],x01[-1,0],color='k',linestyle='--',lw=3)
#plt.xscale('log')
ax01.set_xlim(0,x01[-1,0])
ax01.set_ylim(-0.005,0.055)
ax01.set_xlabel(r'$T~[\mu_{0}^{-1}]$')
ax01.set_ylabel(r'$\dot{\mathcal{T}}_{X_{2}\to X_{1}}~[\mathrm{nats}~\mu_{0}]$')

ax10.errorbar(x10[:,0],x10[:,1],yerr=x10[:,2],marker='^',color='g',ms=12,fillstyle='none',mew=2,linestyle='None')
ax10.hlines(Txy(),x10[0,0],x10[-1,0],color='k',linestyle='--',lw=3)
#plt.xscale('log')
ax10.set_xlim(0,30)
ax10.set_ylim(0.1,0.25)
ax10.set_xlabel(r'$T~[a_{11}^{-1}]$')
ax10.set_ylabel(r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}~[\mathrm{nats}~a_{11}]$')

ax11.errorbar(x11[:,0],x11[:,1],yerr=x11[:,2],marker='x',color='orange',ms=12,fillstyle='none',mew=2,linestyle='None')
ax11.hlines(Tyx(),x11[0,0],x11[-1,0],color='k',linestyle='--',lw=3)
#plt.xscale('log')
ax11.set_xlim(0,30)
ax11.set_ylim(0,0.025)
ax11.set_xlabel(r'$T~[a_{11}^{-1}]$')
ax11.set_ylabel(r'$\dot{\mathcal{T}}_{X_{2}\to X_{1}}~[\mathrm{nats}~a_{11}]$')

#plt.show()
ax00.annotate(r'$(a)$',xy=(-0.12,1.1),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.15,1.1),xycoords='axes fraction')
ax10.annotate(r'$(c)$',xy=(-0.12,1.1),xycoords='axes fraction')
ax11.annotate(r'$(d)$',xy=(-0.15,1.1),xycoords='axes fraction')
plt.savefig('fig_validation.png',bbox_inches='tight',pad_inches=0.1)

