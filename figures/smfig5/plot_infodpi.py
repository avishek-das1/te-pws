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
plt.rc('mathtext',rm='dejavusans')
plt.rc('mathtext',fontset='dejavusans')

fig = plt.figure(figsize = (30, 7))
gs= fig.add_gridspec(1,3,top=0.95,hspace=0.,wspace=0.25)

ax00=fig.add_subplot(gs[0,0])
ax01=fig.add_subplot(gs[0,1])
ax02=fig.add_subplot(gs[0,2])

x0=np.reshape(np.loadtxt('OU_data.txt'),(10,11,18))
x1=np.reshape(np.loadtxt('switching_data.txt'),(10,7,13))

x0m=np.mean(x0,axis=0)
x0e=(np.var(x0,axis=0)/10)**0.5

x1m=np.mean(x1,axis=0)
x1e=(np.var(x1,axis=0)/10)**0.5

ax00.errorbar(x0m[:,1]/0.2,x0m[:,-3],yerr=2*x0e[:,-3],marker='x',color='r',ms=20,fillstyle='none',mew=2.5,linestyle='None',capsize=5,label=r'$\dot{I}(X_{1};X_{2})$')
ax00.errorbar(x0m[:,1]/0.2,x0m[:,-1],yerr=2*x0e[:,-1],marker='s',color='r',ms=20,fillstyle='none',mew=2.5,linestyle='None',capsize=5,label=r'$\dot{I}(X_{1};X_{3})$')
ax01.errorbar(x1m[:,1]/x1m[:,0],x1m[:,4],yerr=2*x1e[:,4],marker='x',color='b',ms=20,fillstyle='none',mew=2.5,linestyle='None',capsize=5,label=r'$\dot{I}(X_{1};X_{2})$')
ax01.errorbar(x1m[:,1]/x1m[:,0],x1m[:,5],yerr=2*x1e[:,5],marker='s',color='b',ms=20,fillstyle='none',mew=2.5,linestyle='None',capsize=5,label=r'$\dot{I}(X_{1};X_{3})$')

ax02.hlines(1,0,15,color='gray',linestyle='--',lw=2)
ax02.errorbar(x0m[:,1]/0.2,x0m[:,-1]/x0m[:,-3],yerr=x0m[:,-1]/x0m[:,-3]*(x0e[:,-1]**2/x0m[:,-1]**2+x0e[:,-3]**2/x0m[:,-3]**2)**0.5,marker='o',color='r',ms=20,fillstyle='none',mew=2.5,linestyle='None',label=r'Model C')
ax02.errorbar(x1m[:,1]/x1m[:,0],x1m[:,5]/x1m[:,4],yerr=x1m[:,5]/x1m[:,4]*(x1e[:,5]**2/x1m[:,5]**2+x1e[:,4]**2/x1m[:,4]**2)**0.5,marker='^',color='b',ms=20,fillstyle='none',mew=2.5,linestyle='None',label=r'Model D')

ax02.set_xlim(0,2.1)
ax02.set_xlabel(r'$f^{~*}$')
ax02.set_ylabel(r'$I^{*}=\dot{I}(X_{1};X_{3})/\dot{I}(X_{1};X_{2})$')

ax02.legend(frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.05, 0.97),handletextpad=0.0)

ax00.set_xlabel(r'$f^{~*}$')
ax00.set_ylabel(r'$\dot{I}~[\mathrm{nats~}a_{11}]$')
ax01.set_xlabel(r'$f^{~*}$')
ax01.set_ylabel(r'$\dot{I}~[\mathrm{nats~}a_{11}]$')
ax00.legend(title=r'$\mathrm{Model~C}$',frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.05, 1.0),handletextpad=0.0)
ax01.legend(title=r'$\mathrm{Model~D}$',frameon='False',framealpha=0.0,loc='upper left',bbox_to_anchor=(-0.05, 1.0),handletextpad=0.0)

ax00.annotate(r'$(a)$',xy=(-0.25,0.97),xycoords='axes fraction')
ax01.annotate(r'$(b)$',xy=(-0.24,0.97),xycoords='axes fraction')
ax02.annotate(r'$(c)$',xy=(-0.22,0.97),xycoords='axes fraction')
plt.savefig('fig_infodpi.png',bbox_inches='tight',pad_inches=0.1)
#plt.show()
