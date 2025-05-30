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
#plt.rc('font', size = 30)
#plt.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('font', size = 28)
#plt.rc('text', usetex=True)
##mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

fig = plt.figure(figsize = (22, 13))
gs= fig.add_gridspec(2,2,top=0.95,hspace=0.5,wspace=0.3)

ax00=fig.add_subplot(gs[0,0])
ax01=fig.add_subplot(gs[0,1])
ax10=fig.add_subplot(gs[1,0])
ax11=fig.add_subplot(gs[1,1])

dt=0.01
nc=60

x=np.loadtxt('data/KSG_a6_k5_subsamplestudy.txt',skiprows=1)
ax00.errorbar(x[:,0],x[:,1],yerr=x[:,2],marker='D',fillstyle='none',color='darkorange',ms=20,mew=2,linestyle='None',capsize=5)
ax00.errorbar(x[-1,0],x[-1,1],yerr=x[-1,2],marker='s',color='brown',ms=25,mew=2,linestyle='None',capsize=5)
ax00.errorbar(x[8,0],x[8,1],yerr=x[8,2],marker='X',color='brown',ms=25,mew=2,linestyle='None',capsize=5)
ax00.errorbar(x[5,0],x[5,1],yerr=x[5,2],marker='^',color='brown',ms=25,mew=2,linestyle='None',capsize=5)

ax00.set_xscale('log')

axins = inset_axes(ax00, width='50%',height='40%',bbox_to_anchor=(400,640,600,500))
axins.errorbar(x[1:-3,0],x[1:-3,1],yerr=x[1:-3,2],marker='D',fillstyle='none',color='darkorange',ms=20,mew=2,linestyle='None',capsize=5)
axins.errorbar(x[8,0],x[8,1],yerr=x[8,2],marker='X',color='brown',ms=25,mew=2,linestyle='None',capsize=5)
axins.errorbar(x[5,0],x[5,1],yerr=x[5,2],marker='^',color='brown',ms=25,mew=2,linestyle='None',capsize=5)

ax00.vlines(1,-0.27,x[-1,1],color='grey',linestyle='--')
ax00.vlines(50,-0.27,x[8,1],color='grey',linestyle='--')
ax00.vlines(100,-0.27,x[5,1],color='grey',linestyle='--')

coords = ax00.transAxes.inverted().transform(axins.get_tightbbox(renderer=fig.canvas.get_renderer()))
border = 0.02
w, h = coords[1] - coords[0] + 2*border
ax00.add_patch(plt.Rectangle(coords[0]-border, w, h, fc="white",
                           transform=ax00.transAxes, zorder=2))

x=np.reshape(np.loadtxt('data/KSG_a6_kvar_downsampling1.txt',skiprows=2)[:-1,:],(4,5,3))
ax01.errorbar(x[0,:-1,0]*dt/nc*1e4,x[0,:-1,1],yerr=x[0,:-1,2],marker='s',color='r',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax01.errorbar(x[1,:-1,0]*dt/nc*1e4,x[1,:-1,1],yerr=x[1,:-1,2],marker='s',color='b',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax01.errorbar(x[2,:-1,0]*dt/nc*1e4,x[2,:-1,1],yerr=x[2,:-1,2],marker='s',color='g',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax01.errorbar(x[3,:-1,0]*dt/nc*1e4,x[3,:-1,1],yerr=x[3,:-1,2],marker='s',color='k',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')

ax01.plot(x[0,:,0]*dt/nc*1e4,x[0,:,1],marker='None',color='r',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')
ax01.plot(x[1,:,0]*dt/nc*1e4,x[1,:,1],marker='None',color='b',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')
ax01.plot(x[2,:,0]*dt/nc*1e4,x[2,:,1],marker='None',color='g',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')
ax01.plot(x[3,:,0]*dt/nc*1e4,x[3,:,1],marker='None',color='k',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')

#ax01.set_xticks([y[0,0]*dt/nc,y[-1,0]*dt/nc])

ax01.errorbar(x[0,-1,0]*dt/nc*1e4,x[0,-1,1],yerr=x[0,-1,2],marker='*',color='r',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax01.errorbar(x[1,-1,0]*dt/nc*1e4,x[1,-1,1],yerr=x[1,-1,2],marker='*',color='b',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax01.errorbar(x[2,-1,0]*dt/nc*1e4,x[2,-1,1],yerr=x[2,-1,2],marker='*',color='g',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax01.errorbar(x[3,-1,0]*dt/nc*1e4,x[3,-1,1],yerr=x[3,-1,2],marker='*',color='k',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')

################################################################
axins2=inset_axes(ax01, width='50%',height='40%',bbox_to_anchor=(1050,750,600,500))
axins2.errorbar(1/(1600*1000/5/3)*1e6,x[0,-1,1],yerr=x[0,-1,2],marker='*',color='r',ms=25,mew=2,linestyle='None',capsize=5)
axins2.errorbar(1/(1600*1000/5/2)*1e6,x[1,-1,1],yerr=x[1,-1,2],marker='*',color='b',ms=25,mew=2,linestyle='None',capsize=5)
axins2.errorbar(1/(1600*1000/5/1)*1e6,x[2,-1,1],yerr=x[2,-1,2],marker='*',color='g',ms=25,mew=2,linestyle='None',capsize=5)
axins2.errorbar(0,x[-1,-1,1],yerr=x[-1,-1,2],marker='*',color='k',ms=25,mew=2,linestyle='None',capsize=5)

intercept=x[-1,-1,1]
ierr=x[-1,-1,2]
slope=1.02289892e+05
serr=7.89906243e+03
M=50
newN=np.linspace(0,1/(1600*1000/5/3),M)
lmax=(slope+serr)*newN+(intercept+ierr)
lmin=(slope-serr)*newN+(intercept-ierr)
l0=slope*newN+intercept
axins2.plot(newN*1e6,l0,color='k',lw=2)
axins2.fill_between(newN*1e6,lmin,lmax,color='k',alpha=0.1)

###########################################################
x=np.reshape(np.loadtxt('data/KSG_a6_kvar_downsampling50.txt',skiprows=2)[:-1,:],(4,10,3))
ax10.errorbar(x[0,:-1,0]*dt/nc,x[0,:-1,1],yerr=x[0,:-1,2],marker='s',color='r',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax10.errorbar(x[1,:-1,0]*dt/nc,x[1,:-1,1],yerr=x[1,:-1,2],marker='s',color='b',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax10.errorbar(x[2,:-1,0]*dt/nc,x[2,:-1,1],yerr=x[2,:-1,2],marker='s',color='g',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax10.errorbar(x[3,:-1,0]*dt/nc,x[3,:-1,1],yerr=x[3,:-1,2],marker='s',color='k',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')

ax10.plot(x[0,:,0]*dt/nc,x[0,:,1],marker='None',color='r',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')
ax10.plot(x[1,:,0]*dt/nc,x[1,:,1],marker='None',color='b',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')
ax10.plot(x[2,:,0]*dt/nc,x[2,:,1],marker='None',color='g',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')
ax10.plot(x[3,:,0]*dt/nc,x[3,:,1],marker='None',color='k',ms=20,fillstyle='none',mew=2,linestyle='-',label=r'$M_{1}=2000$')

ax10.errorbar(x[0,-1,0]*dt/nc,x[0,-1,1],yerr=x[0,-1,2],marker='*',color='r',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax10.errorbar(x[1,-1,0]*dt/nc,x[1,-1,1],yerr=x[1,-1,2],marker='*',color='b',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax10.errorbar(x[2,-1,0]*dt/nc,x[2,-1,1],yerr=x[2,-1,2],marker='*',color='g',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax10.errorbar(x[3,-1,0]*dt/nc,x[3,-1,1],yerr=x[3,-1,2],marker='*',color='k',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')

################################################################
axins3=inset_axes(ax10, width='50%',height='40%',bbox_to_anchor=(250,-25,600,500))
axins3.errorbar(1/(1600*1000*4/50/3)*1e5,x[0,-1,1],yerr=x[0,-1,2],marker='*',color='r',ms=25,mew=2,linestyle='None',capsize=5)
axins3.errorbar(1/(1600*1000*4/50/2)*1e5,x[1,-1,1],yerr=x[1,-1,2],marker='*',color='b',ms=25,mew=2,linestyle='None',capsize=5)
axins3.errorbar(1/(1600*1000*4/50/1)*1e5,x[2,-1,1],yerr=x[2,-1,2],marker='*',color='g',ms=25,mew=2,linestyle='None',capsize=5)
axins3.errorbar(0,x[-1,-1,1],yerr=x[-1,-1,2],marker='*',color='k',ms=25,mew=2,linestyle='None',capsize=5)

intercept=x[-1,-1,1]
ierr=x[-1,-1,2]
slope=-4.00679386e+02
serr=2.58841883e+01
M=50
newN=np.linspace(0,1/(1600*1000*4/50/3),M)
lmax=(slope+serr)*newN+(intercept+ierr)
lmin=(slope-serr)*newN+(intercept-ierr)
l0=slope*newN+intercept
axins3.plot(newN*1e5,l0,color='k',lw=2)
axins3.fill_between(newN*1e5,lmin,lmax,color='k',alpha=0.1)
###########################################################

x=np.reshape(np.loadtxt('data/KSG_a6_kvar_downsampling100.txt',skiprows=2)[:-1,:],(4,10,3))
ax11.errorbar(x[0,:-1,0]*dt/nc,x[0,:-1,1],yerr=x[0,:-1,2],marker='s',color='r',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[1,:-1,0]*dt/nc,x[1,:-1,1],yerr=x[1,:-1,2],marker='s',color='b',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[2,:-1,0]*dt/nc,x[2,:-1,1],yerr=x[2,:-1,2],marker='s',color='g',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[3,:-1,0]*dt/nc,x[3,:-1,1],yerr=x[3,:-1,2],marker='s',color='k',ms=20,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')

ax11.errorbar(x[0,:,0]*dt/nc,x[0,:,1],yerr=x[0,:,2],marker='None',color='r',ms=30,mew=2,linestyle='-',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[1,:,0]*dt/nc,x[1,:,1],yerr=x[1,:,2],marker='None',color='b',ms=30,mew=2,linestyle='-',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[2,:,0]*dt/nc,x[2,:,1],yerr=x[2,:,2],marker='None',color='g',ms=30,mew=2,linestyle='-',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[3,:,0]*dt/nc,x[3,:,1],yerr=x[3,:,2],marker='None',color='k',ms=30,mew=2,linestyle='-',capsize=5,label=r'$M_{1}=2000$')

ax11.errorbar(x[0,-1,0]*dt/nc,x[0,-1,1],yerr=x[0,-1,2],marker='*',color='r',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[1,-1,0]*dt/nc,x[1,-1,1],yerr=x[1,-1,2],marker='*',color='b',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[2,-1,0]*dt/nc,x[2,-1,1],yerr=x[2,-1,2],marker='*',color='g',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')
ax11.errorbar(x[3,-1,0]*dt/nc,x[3,-1,1],yerr=x[3,-1,2],marker='*',color='k',ms=30,mew=2,linestyle='None',capsize=5,label=r'$M_{1}=2000$')

################################################################
axins4=inset_axes(ax11, width='50%',height='40%',bbox_to_anchor=(1200,50,600,500))
axins4.errorbar(1/(1600*1000*8/100/3)*1e5,x[0,-1,1],yerr=x[0,-1,2],marker='*',color='r',ms=25,mew=2,linestyle='None',capsize=5)
axins4.errorbar(1/(1600*1000*8/100/2)*1e5,x[1,-1,1],yerr=x[1,-1,2],marker='*',color='b',ms=25,mew=2,linestyle='None',capsize=5)
axins4.errorbar(1/(1600*1000*8/100/1)*1e5,x[2,-1,1],yerr=x[2,-1,2],marker='*',color='g',ms=25,mew=2,linestyle='None',capsize=5)
axins4.errorbar(0,x[-1,-1,1],yerr=x[-1,-1,2],marker='*',color='k',ms=25,mew=2,linestyle='None',capsize=5)

intercept=x[-1,-1,1]
ierr=x[-1,-1,2]
slope=11.4922798
serr=2.26521032e+01
M=50
newN=np.linspace(0,1/(1600*1000*8/100/3),M)
lmax=(slope+serr)*newN+(intercept+ierr)
lmin=(slope-serr)*newN+(intercept-ierr)
l0=slope*newN+intercept
axins4.plot(newN*1e5,l0,color='k',lw=2)
axins4.fill_between(newN*1e5,lmin,lmax,color='k',alpha=0.1)
#########################################################################

a1 = mpl.patches.FancyArrowPatch((4.5*dt/nc*1e4, 0.7), (4.5*dt/nc*1e4, -0.4),
        connectionstyle="arc3,rad=0.0",color='darkorange',alpha=0.9,
        arrowstyle="fancy,head_width=20,head_length=20,tail_width=10",
        lw=2,clip_on=False,zorder=11)
ax01.add_patch(a1)

a1 = mpl.patches.FancyArrowPatch((7000*dt/nc, 0.04), (7000*dt/nc, 0.065),
        connectionstyle="arc3,rad=0.0",color='darkorange',alpha=0.9,
        arrowstyle="fancy,head_width=20,head_length=20,tail_width=10",
        lw=2,clip_on=False,zorder=11)
ax10.add_patch(a1)

a1 = mpl.patches.FancyArrowPatch((7000*dt/nc, 0.028), (7000*dt/nc, 0.02),
        connectionstyle="arc3,rad=0.0",color='darkorange',alpha=0.9,
        arrowstyle="fancy,head_width=20,head_length=20,tail_width=10",
        lw=2,clip_on=False,zorder=11)
ax11.add_patch(a1)

############################################################

ax00.set_ylabel(r'$\dot{\mathcal{T}}^{~[k]}_{X_{1}\to X_{3}}[\mathrm{nats~}a_{11}]$')
ax00.set_xlabel(r"$\mathrm{Downsampling~rate~}\delta t/\delta t_{0}$")
ax01.set_ylabel(r'$\dot{\mathcal{T}}^{~[k]}_{X_{1}\to X_{3}}[\mathrm{nats~}a_{11}]$')
ax01.set_xlabel(r'$[(k+1)\delta t_{1}/\tau].10^{4}$')
ax10.set_ylabel(r'$\dot{\mathcal{T}}^{~[k]}_{X_{1}\to X_{3}}[\mathrm{nats~}a_{11}]$')
ax10.set_xlabel(r'$(k+1)\delta t_{2}/\tau$')
ax11.set_ylabel(r'$\dot{\mathcal{T}}^{~[k]}_{X_{1}\to X_{3}}[\mathrm{nats~}a_{11}]$')
ax11.set_xlabel(r'$(k+1)\delta t_{3}/\tau$')
axins2.set_xlabel(r'$1/(NM_{1}.10^{-6})$')
axins3.set_xlabel(r'$1/(NM_{1}.10^{-5})$')
axins4.set_xlabel(r'$1/(NM_{1}.10^{-5})$')
#axins2.get_xaxis().get_offset_text().set_position((1.2,0))
#axins3.get_xaxis().get_offset_text().set_position((1.2,0))
#axins4.get_xaxis().get_offset_text().set_position((1.2,0))

ax00.set_title(r'$k+1=5,~NM_{1}\delta t/\delta t_{0}=1.6\times 10^{6},~\mathrm{varying~}\delta t$',fontsize=28,pad=15)
ax01.set_title(r'$\delta t=\delta t_{1}=\delta t_{0},~\mathrm{varying}~k~\mathrm{and~}NM_{1}$',fontsize=28,pad=15)
ax10.set_title(r'$\delta t=\delta t_{2}=50\delta t_{0},~\mathrm{varying}~k~\mathrm{and~}NM_{1}$',fontsize=28,pad=15)
ax11.set_title(r'$\delta t=\delta t_{3}=100\delta t_{0},~\mathrm{varying}~k~\mathrm{and~}NM_{1}$',fontsize=28,pad=15)
#axins2.set_ylabel(r'$\dot{\mathcal{T}}^{[\kappa\to\infty]}_{X_{1}\to X_{3}}$')
#axins2.set_xlabel(r'$1/(NM_{1})$')
#axins3.set_ylabel(r'$\dot{\mathcal{T}}^{[\kappa\to\infty]}_{X_{1}\to X_{3}}$')
#axins3.set_xlabel(r'$1/M$')
#axins4.set_ylabel(r'$\dot{\mathcal{T}}^{[\kappa\to\infty]}_{X_{1}\to X_{3}}$')
#axins4.set_xlabel(r'$1/M$')
#ax11.set_ylim(0.026,0.06)
ax00.set_ylim(-0.25,0.055)
ax01.set_ylim(-0.5,1.5)
ax10.set_ylim(-0.04,0.08)
ax11.set_ylim(0.018,0.05)

ax01.set_xlim(0.0001*1e4,5.5*dt/nc*1e4)
ax10.set_xlim(0,8200*dt/nc)
ax11.set_xlim(0,8200*dt/nc)

ax00.annotate(r'$\delta t_{1}$',xy=(1.1,-0.24),xycoords='data')
ax00.annotate(r'$\delta t_{2}$',xy=(27,-0.24),xycoords='data')
ax00.annotate(r'$\delta t_{3}$',xy=(105,-0.24),xycoords='data')
ax00.set_xticks([1,50,100])
ax00.set_xticklabels(labels=[r'$1$',r'$50$',r'$100$'])
axins.set_xticks([0,150,300])

ax01.ticklabel_format(axis='x', style='scientific',scilimits=[0,0])
#ax01.set_xticks([1.7,8.3])
ax10.ticklabel_format(axis='x', style='scientific',scilimits=[0,0])
#ax10.set_xticks([4.2e-1,75e-1])
ax11.ticklabel_format(axis='x', style='scientific',scilimits=[0,0])
#ax11.set_xticks([0.0083*10,0.15*10])
#axins2.ticklabel_format(axis='x',style='sci')
#axins3.ticklabel_format(axis='x',style='sci')

#ax01.xaxis.set_label_coords(1.6,-0.12)

ax00.annotate(r'$(a)$',xy=(-0.24,0.95),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.24,0.95),xycoords='axes fraction')
ax10.annotate(r'$(c)$',xy=(-0.24,0.95),xycoords='axes fraction')
ax11.annotate(r'$(d)$',xy=(-0.24,0.95),xycoords='axes fraction')
plt.savefig('fig_convg_KSG.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()
