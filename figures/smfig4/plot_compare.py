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
import os.path
from scipy.optimize import curve_fit

#mpl.style.use("classic")
#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#mpl.rcParams.update({'font.size': 28})
plt.rc('font', size = 28)
#plt.rc('text', usetex=True)
##mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

#fig = plt.figure(figsize = (22, 8))
fig = plt.figure(figsize = (22, 7))
plt.subplots_adjust(top=0.95,hspace=0.2,wspace=0.28)
gs = gridspec.GridSpec(4, 2)

axn0=fig.add_subplot(gs[0,0])
axn1=fig.add_subplot(gs[0,1])
ax00=fig.add_subplot(gs[1:,0])
ax01=fig.add_subplot(gs[1:,1])

#axins0 = inset_axes(ax00, width='30%',height='30%',bbox_to_anchor=(100,50,750,500))

dt=0.01
nc0=2
nc1=60

x0l=0.1
x0h=6400*dt/nc0

x0g=np.reshape(np.loadtxt('data/Gaussian_OU_fullhist.txt',skiprows=2),(4,19,3))[-1,:,:]
pg=ax00.errorbar(x0g[:,0]*dt/nc0,x0g[:,1],yerr=x0g[:,2],marker='o',color='b',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{Gaussian}$')

x0k=np.loadtxt('data/KSG_OU_kvar_downsampling40.txt',skiprows=2)
pk=ax00.errorbar(x0k[:,0]*dt*40/nc0,x0k[:,1],yerr=x0k[:,2],marker='^',color='r',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{KSG~}(k\to\infty)$')

x0k1=np.loadtxt('data/KSG_OU_k1.txt',skiprows=4)
pk1=ax00.hlines(x0k1[3,0],x0l,x0h,lw=2,color='brown',label=r'$\mathrm{KSG~}(k=0)$')
ax00.fill_between(np.array([x0l,x0h]),np.array([x0k1[3,0]-x0k1[3,1]]),np.array([x0k1[3,0]+x0k1[3,1]]),color='brown',alpha=0.3)
pk1n=ax00.fill(np.NaN,np.NaN,color='brown',alpha=0.3)

x0t=np.loadtxt('data/TEPWS_OU.txt',skiprows=1)[:-1,:]
pt=ax00.plot(x0t[:,0]*dt/nc0,x0t[:,1],marker='None',color='k',ms=20,fillstyle='none',mew=2,linestyle='-',lw=2,label=r'$\mathrm{Exact}$')
ax00.fill_between(x0t[:,0]*dt/nc0,x0t[:,1]-x0t[:,2],x0t[:,1]+x0t[:,2],color='k',alpha=0.4)
ptn=ax00.fill(np.NaN,np.NaN,color='k',alpha=0.4)

ax00.set_xlim(x0l,x0h)
ax00.set_ylim(0,0.14)

x1l=0.01
x1h=8000*dt/nc1
x1g=np.reshape(np.loadtxt('data/Gaussian_a6_fullhist.txt',skiprows=2),(4,29,3))[-1,:,:]
ax01.errorbar(x1g[:,0]*dt/nc1,x1g[:,1],yerr=x1g[:,2],marker='o',color='b',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{Gaussian}$')

x1k=np.loadtxt('data/KSG_a6_kvar_downsampling100.txt',skiprows=3)[-11:-1,:]
ax01.errorbar(x1k[:,0]*dt/nc1,x1k[:,1],yerr=x1k[:,2],marker='^',color='r',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{KSG~(k\to\infty)}$')

x1k1=np.loadtxt('data/KSG_a6_k1.txt',skiprows=4)
#ax00.errorbar(1*dt/nc0,x0k1[3,0],yerr=x0k1[3,1],marker='D',color='brown',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{KSG~(k=0)}$')
ax01.hlines(x1k1[3,0],x1l,x1h,lw=3,color='brown')
ax01.fill_between(np.array([x1l,x1h]),np.array([x1k1[3,0]-x1k1[3,1]]),np.array([x1k1[3,0]+x1k1[3,1]]),color='brown',alpha=0.3)

x1t=np.loadtxt('data/TEPWS_a6.txt',skiprows=1)[:-1,:]
ax01.plot(x1t[:,0]*dt/nc1,x1t[:,1],marker='None',color='k',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$\mathrm{Exact}$')
ax01.fill_between(x1t[:,0]*dt/nc1,x1t[:,1]-x1t[:,2],x1t[:,1]+x1t[:,2],color='k',alpha=0.4)

ax01.set_xlim(x1l,x1h)
ax01.set_ylim(0.01,0.09)

ax00.set_xlabel(r'$\mathrm{History~length~}(k+1)\delta t/\tau$')
ax00.set_ylabel(r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}~[\mathrm{nats}~a_{11}]$')
ax01.set_xlabel(r'$\mathrm{History~length~}(k+1)\delta t/\tau$')
ax01.set_ylabel(r'$\dot{\mathcal{T}}_{X_{1}\to X_{3}}~[\mathrm{nats}~a_{11}]$')

#plt.tight_layout()
ax00.set_xscale('log')
ax01.set_xscale('log')
#ax00.set_xticks([0.001,0.01,0.1,1])
#plt.show()

oss1=500
oss2=500
t66=np.loadtxt('data/trajxy_OU.txt',skiprows=1)
t44=np.loadtxt('data/trajxz_a6.txt',skiprows=1)
axn0.plot(t66[500+oss1:1000+oss1,0],color='g')
axn0.plot(t66[500+oss1:1000+oss1,1],color='orange')
axn1.plot(t44[500+oss2:1000+oss2,0],color='g')
axn1.plot(t44[500+oss2:1000+oss2,1],color='orange')
axn1.set_xlim(0,499)
axn0.set_xlim(0,499)
axn1.set_ylim(-4,22)
axn0.set_ylim(-6,14)

axn0.set_xticks([])
axn0.set_yticks([])
axn1.set_xticks([])
axn1.set_yticks([])

axn1.arrow(160,10,150,0,lw=1.5,color='k',head_width=2.4,head_length=10,length_includes_head=True,zorder=4)
axn1.arrow(310,10,-150,0,lw=1.5,color='k',head_width=2.4,head_length=10,length_includes_head=True,zorder=4)
axn1.annotate(r'$\tau\approx 60a_{11}^{-1}$',xy=(0.33,0.7),xycoords='axes fraction')
axn0.arrow(130,6,50,0,lw=1.5,color='k',head_width=2.4,head_length=10,length_includes_head=True,zorder=4)
axn0.arrow(180,6,-50,0,lw=1.5,color='k',head_width=2.4,head_length=10,length_includes_head=True,zorder=4)
axn0.annotate(r'$\tau\approx 2a_{11}^{-1}$',xy=(0.28,0.7),xycoords='axes fraction')

ax00.legend([(pt[0],ptn[0]),(pk1,pk1n[0]),pg[0],pk[0],],[r'$\mathrm{Exact}$',r'$\mathrm{KSG}~(k=0)$',r'$\mathrm{Gaussian}$',r'$\mathrm{KSG}~(k\to\infty)$'],frameon='False',framealpha=0.0,loc='upper right',bbox_to_anchor=(2.85, 0.95),handletextpad=1)

ax00.xaxis.set_tick_params(pad=10)
ax01.xaxis.set_tick_params(pad=10)

ax00.annotate(r'$(a)$',xy=(-0.1,1.3),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.23,1.3),xycoords='axes fraction')
ax00.annotate(r'$(c)$',xy=(-0.1,0.93),xycoords='axes fraction')
ax01.annotate(r'$(d)$',xy=(-0.23,0.93),xycoords='axes fraction')
ax00.annotate(r'Linear model (model A)',xy=(0.18,1.42),xycoords='axes fraction')
ax01.annotate(r'Nonlinear model (model D)',xy=(0.16,1.42),xycoords='axes fraction')
plt.savefig('fig_compare.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()
