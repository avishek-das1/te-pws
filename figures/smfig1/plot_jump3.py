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

plt.rc('font', size = 28)
#plt.rc('text', usetex=True)
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

fig = plt.figure(figsize = (22, 7))
gs= fig.add_gridspec(1,2,top=0.95,hspace=0.,wspace=0.23)

ax00=fig.add_subplot(gs[0,0])
ax01=fig.add_subplot(gs[0,1])

dt=0.005 #for checking resampling and for computing TE
#when increasing dt, always increase Mmin
M=40
stepmags=np.arange(1,1+M,1)#np.arange(1,1001,1)
ttot=stepmags*dt

N=160 #number of realizations
cnt=0
MM=2 #plot once every MM steps

Iestxy=np.zeros((N,M))
Tvalsxy=np.zeros((N,M))
Twvalsxy=np.zeros((N,M))

Ime=np.zeros((M,2))
Tme=np.zeros((M,2))
Twme=np.zeros((M,2))

for i in range(N):
    Iestxy[cnt,:]=np.loadtxt('data/Iestxy_'+str(i)+'.txt')
    Tvalsxy[cnt,:]=np.loadtxt('data/Tvalsxy_'+str(i)+'.txt') #jump-based estimate
    Twvalsxy[cnt,:]=np.loadtxt('data/Twvalsxy_'+str(i)+'.txt') #reduced variance estimate
    cnt+=1
Ime[:,0]=np.mean(Iestxy[:cnt],axis=0)
Ime[:,1]=2*(np.var(Iestxy[:cnt],axis=0)/cnt)**0.5
Tme[:,0]=np.mean(Tvalsxy[:cnt],axis=0)
Tme[:,1]=2*(np.var(Tvalsxy[:cnt],axis=0)/cnt)**0.5
Twme[:,0]=np.mean(Twvalsxy[:cnt],axis=0)
Twme[:,1]=2*(np.var(Twvalsxy[:cnt],axis=0)/cnt)**0.5

gtruth=Ime[-1,0]

ax00.errorbar(ttot[::MM],Tme[::MM,0],yerr=Tme[::MM,1],marker='o',color='r',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathcal{T}_{X\to Y}^{~(\mathrm{J})}$')
ax00.errorbar(ttot[::MM],Twme[::MM,0],yerr=Twme[::MM,1],marker='^',color='b',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathcal{T}_{X\to Y}^{~(\mathrm{J}+\mathrm{E})}$')
ax00.plot(ttot[::MM],Ime[::MM,0],marker='None',color='k',ms=10,fillstyle='none',mew=2.5,linestyle='--',lw=2,label=r'$\mathrm{PWS}$')

ax00.set_xlabel(r'$T~[1/k_{-1}]$')
ax00.set_ylabel(r'$\mathcal{T}_{X\to Y}~[\mathrm{nats}]$')
ax00.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.02, 1.02))

P=5
Ns=np.array([10,20,40,80,160])
Tmes=np.zeros((P,2))
Twmes=np.zeros((P,2))
for p in range(P):
    cnt=int(round(Ns[p]))
    Tmes[p,0]=np.mean((Tvalsxy[:cnt,-1]-gtruth))
    Tmes[p,1]=2.*(np.var((Tvalsxy[:cnt,-1]-gtruth))/cnt)**0.5
    Twmes[p,0]=np.mean((Twvalsxy[:cnt,-1]-gtruth))
    Twmes[p,1]=2.*(np.var((Twvalsxy[:cnt,-1]-gtruth))/cnt)**0.5

Ns*=160
ax01.errorbar(Ns,Tmes[:,0],yerr=Tmes[:,1],marker='s',color='darkorange',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{Error~in~}\mathcal{T}_{X\to Y}^{~(\mathrm{J})}(T=0.2)$')
ax01.errorbar(Ns,Twmes[:,0],yerr=Twmes[:,1],marker='x',color='green',ms=15,fillstyle='none',mew=2,linestyle='None',capsize=5,label=r'$\mathrm{Error~in~}\mathcal{T}_{X\to Y}^{~(\mathrm{J}+\mathrm{E})}(T=0.2)$')
ax01.set_xlabel(r'$M_{1}$')
ax01.set_ylabel(r'$\mathrm{Error~in~}\mathcal{T}_{X\to Y}(T=0.2)~[\mathrm{nats}]$')
ax01.set_xscale('log')
ax01.set_xticks([2000,10000])
ax01.set_xticklabels([r'$2000$',r'$10000$'])
ax01.set_ylim(-0.3,0.4)
ax01.legend(frameon='False',framealpha=0.0,loc='upper right',bbox_to_anchor=(1.02, 1.02),handletextpad=0.0)

#plt.show()
ax00.annotate(r'$(a)$',xy=(-0.17,1.0),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.17,1.0),xycoords='axes fraction')
plt.savefig('fig_jump3.png',bbox_inches='tight',pad_inches=0.1)

