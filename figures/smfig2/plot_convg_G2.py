import math
import copy
import time
import numba
import numpy as np
import scipy.ndimage
import numpy.random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import ticker
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
import os.path

#mpl.style.use("classic")
#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#mpl.rcParams.update({'font.size': 28})
plt.rc('font', size = 28)
#plt.rc('text', usetex=True)
##mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

fig = plt.figure(figsize = (22, 7))
gs= fig.add_gridspec(1,2,top=0.95,hspace=0.3,wspace=0.3)

ax00=fig.add_subplot(gs[0,0])
ax01=fig.add_subplot(gs[0,1])

dt=0.01
nc=60
nc2=11

Ns=np.array([1/1.6e6,2/1.6e6,3/1.6e6])

###a=6#######################################################################################
xs=np.reshape(np.loadtxt('fullhist.txt',skiprows=2),(4,29,3))

ax00.errorbar(xs[0,:,0]*dt/nc,xs[0,:,1],yerr=xs[0,:,2],marker='o',color='r',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$NM_{1}=1.6\times 10^{6}/3$')
ax00.errorbar(xs[1,:,0]*dt/nc,xs[1,:,1],yerr=xs[1,:,2],marker='o',color='b',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$NM_{1}=1.6\times 10^{6}/2$')
ax00.errorbar(xs[2,:,0]*dt/nc,xs[2,:,1],yerr=xs[2,:,2],marker='o',color='g',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$NM_{1}=1.6\times 10^{6}$')
ax00.errorbar(xs[3,:,0]*dt/nc,xs[3,:,1],yerr=xs[3,:,2],marker='o',color='k',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$NM_{1}\to\infty$')

ax00.errorbar(xs[0,-1,0]*dt/nc,xs[0,-1,1],yerr=xs[0,-1,2],marker='*',color='r',ms=25,mew=2,linestyle='None',capsize=5)
ax00.errorbar(xs[1,-1,0]*dt/nc,xs[1,-1,1],yerr=xs[1,-1,2],marker='*',color='b',ms=25,mew=2,linestyle='None',capsize=5)
ax00.errorbar(xs[2,-1,0]*dt/nc,xs[2,-1,1],yerr=xs[2,-1,2],marker='*',color='g',ms=25,mew=2,linestyle='None',capsize=5)
ax00.errorbar(xs[3,-1,0]*dt/nc,xs[3,-1,1],yerr=xs[3,-1,2],marker='*',color='k',ms=25,mew=2,linestyle='None',capsize=5)

a1 = mpl.patches.FancyArrowPatch((0.85, 0.6), (1, 0.1),
        connectionstyle="arc3,rad=-0.2",color='k',alpha=0.4,
        arrowstyle="fancy,head_width=20,head_length=20,tail_width=10",
        lw=2,clip_on=False,zorder=11)
ax00.add_patch(a1)

ax01.errorbar(1/1.6e6*1e7,xs[2,-1,1],yerr=xs[2,-1,2],marker='*',color='g',ms=25,mew=2,linestyle='None',capsize=5)
ax01.errorbar(2/1.6e6*1e7,xs[1,-1,1],yerr=xs[1,-1,2],marker='*',color='b',ms=25,mew=2,linestyle='None',capsize=5)
ax01.errorbar(3/1.6e6*1e7,xs[0,-1,1],yerr=xs[0,-1,2],marker='*',color='r',ms=25,mew=2,linestyle='None',capsize=5)
ax01.errorbar(0,xs[3,-1,1],yerr=xs[3,-1,2],marker='*',color='k',ms=25,mew=2,linestyle='None',capsize=5)

#for highest k, slope of linear fit and 2sigma error
slopem=419198.71279494994
slopee=6068.928703750288

M=50
newN=np.linspace(0,Ns[-1],M)
lmax=(slopem+slopee)*newN+(xs[3,-1,1]+xs[3,-1,2])
lmin=(slopem-slopee)*newN+(xs[3,-1,1]-xs[3,-1,2])
l0=slopem*newN+xs[3,-1,1]
ax01.plot(newN*1e7,l0,color='k',lw=2)
ax01.fill_between(newN*1e7,lmin,lmax,color='k',alpha=0.1)

ax00.set_ylabel(r'$\dot{\mathcal{T}}^{~[k]}_{X_{1}\to X_{3}}[\mathrm{nats~}a_{11}]$')
ax01.set_ylabel(r'$\dot{\mathcal{T}}^{~[k\to\infty]}_{X_{1}\to X_{3}}[\mathrm{nats~}a_{11}]$')
ax00.set_xlabel(r'$\mathrm{History~length~}(k+1)\delta t/\tau$')
ax01.set_xlabel(r'$1/(NM_{1}.10^{-7})$')
#ax01.set_xticks([0,1/10000,1/5000,1/2000])
#ax01.set_xticklabels(labels=[r'$0$',r'$10^{-4}$',r'$1/5000$',r'$1/2000$'])
#ax01.set_ylim(0.06,0.09)
#ax00.set_yscale('log')

ax00.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.05, 1.0),handletextpad=0.0)

ax00.annotate(r'$(a)$',xy=(-0.23,0.97),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.25,0.97),xycoords='axes fraction')
plt.savefig('fig_convg_G.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()
