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
plt.rc('font', size = 28)
#plt.rc('text', usetex = True)
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

fig = plt.figure(figsize = (14, 18))
plt.subplots_adjust(hspace=0.7,wspace=0.35)
gs = gridspec.GridSpec(6, 2)
ax00=fig.add_subplot(gs[0,0])
ax01=fig.add_subplot(gs[0,1])
axn0=fig.add_subplot(gs[1:3,0])
axn1=fig.add_subplot(gs[1:3,1])
ax10=fig.add_subplot(gs[3:,:])

xl=np.loadtxt('te_linear.txt',skiprows=1)
xs=np.loadtxt('te_switching.txt',skiprows=1)

axn0.errorbar(xl[:,0]/xl[:,1],xl[:,4],yerr=xl[:,5],marker='s',color='r',ms=15,fillstyle='none',mew=2.5,linestyle='None',label=r'$\dot{\mathcal{T}}_{X_{1}\to X_{3}}^{~(\mathrm{C})}$')
axn0.errorbar(xl[:,0]/xl[:,1],xl[:,2],yerr=xl[:,3],marker='x',color='r',ms=15,fillstyle='none',mew=2.5,linestyle='None',label=r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}^{~(\mathrm{C})}$')

axn1.errorbar(xs[:,0]/xs[:,1],xs[:,4],yerr=xs[:,5],marker='s',color='b',ms=15,fillstyle='none',mew=2.5,linestyle='None',label=r'$\dot{\mathcal{T}}_{X_{1}\to X_{3}}^{~(\mathrm{D})}$')
axn1.errorbar(xs[:,0]/xs[:,1],xs[:,2],yerr=xs[:,3],marker='x',color='b',ms=15,fillstyle='none',mew=2.5,linestyle='None',label=r'$\dot{\mathcal{T}}_{X_{1}\to X_{2}}^{~(\mathrm{D})}$')

axn0.set_xlabel(r'$f^{~*}$')
axn1.set_xlabel(r'$f^{~*}$')
axn0.set_ylabel(r'$\dot{\mathcal{T}}~[\mathrm{nats}~a_{11}]$')
axn1.set_ylabel(r'$\dot{\mathcal{T}}~[\mathrm{nats}~a_{11}]$')
axn0.set_ylim(0.004,0.023)
axn1.set_ylim(0.005,0.13)
axn0.set_yticks([0.01,0.02])
axn1.set_yticks([0.05,0.1])
axn0.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.1, 1.05),handletextpad=0.0)
axn1.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.1, 1.05),handletextpad=0.0)

ax10.hlines(1,0,15,color='gray',linestyle='--',lw=2)
ax10.errorbar(xl[:,0]/xl[:,1],xl[:,4]/xl[:,2],yerr=xl[:,4]/xl[:,2]*(xl[:,5]**2/xl[:,4]**2+xl[:,3]**2/xl[:,2]**2)**0.5,marker='o',color='r',ms=22,fillstyle='none',mew=2.5,linestyle='None',label=r'Model C')
ax10.errorbar(xs[:,0]/xs[:,1],xs[:,4]/xs[:,2],yerr=xs[:,4]/xs[:,2]*(xs[:,5]**2/xs[:,4]**2+xs[:,3]**2/xs[:,2]**2)**0.5,marker='^',color='b',ms=22,fillstyle='none',mew=2.5,linestyle='None',label=r'Model D')
#ax10.set_xscale('log')
ax10.set_xlim(0,2.1)
#ax00.set_ylim(-0.005,0.125)
ax10.set_xlabel(r'$f^{~*}$')
ax10.set_ylabel(r'$\mathcal{T}^{~*}=\dot{\mathcal{T}}_{X_{1}\to X_{3}}/\dot{\mathcal{T}}_{X_{1}\to X_{2}}$')
#plt.tight_layout()
#plt.show()
#np.savetxt('data_experiment_t21.txt',np.stack((ts[::s]*mu0unit,Tn[::s,0]/ts[::s]/mu0unit,Tne[::s,0]/ts[::s]/mu0unit),axis=-1))
ax10.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.02, 1.02),handletextpad=0.0)

xA=np.loadtxt('traj_linear.txt',skiprows=1)
xB=np.loadtxt('traj_switching.txt',skiprows=1)

ax00.plot(xA[:,0],color='#4cb428ff',lw=1)
ax00.plot(xA[:,1],color='k',lw=1)
ax00.set_xlim(0,20000)

ax01.plot(xB[:,0],color='#4cb428ff',lw=1)
ax01.plot(xB[:,1],color='k',lw=1)
ax01.set_xlim(8000,28000)

ax00.set_xticks([])
ax00.set_yticks([])
ax01.set_xticks([])
ax01.set_yticks([])

ax00.set_xlabel(r'$t$')
ax01.set_xlabel(r'$t$')

ax00.annotate(r'$\mathrm{Model~C}$',xy=(0.35,1.1),xycoords='axes fraction')
ax01.annotate(r'$\mathrm{Model~D}$',xy=(0.35,1.1),xycoords='axes fraction')

#plt.show()
ax00.annotate(r'$(a)$',xy=(-0.2,0.9),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.14,0.9),xycoords='axes fraction')
ax00.annotate(r'$(c)$',xy=(-0.2,-0.8),xycoords='axes fraction')
ax01.annotate(r'$(d)$',xy=(-0.14,-0.8),xycoords='axes fraction')
ax00.annotate(r'$(e)$',xy=(-0.2,-4.1),xycoords='axes fraction')
plt.savefig('fig_dpi_4.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()
