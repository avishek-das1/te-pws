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
from matplotlib.patches import Rectangle, FancyBboxPatch

#mpl.style.use("classic")
#mpl.rcParams['mathtext.fontset'] = 'cm'
#mpl.rcParams['mathtext.rm'] = 'serif'
#mpl.rcParams.update({'font.size': 22})
plt.rc('font', size = 46)
plt.rc('text', usetex = True)

fig = plt.figure(figsize = (10, 12))
plt.subplots_adjust(hspace=0.5,wspace=0.3)
gs = gridspec.GridSpec(1, 1)
ax00=fig.add_subplot(gs[0,0])

schmn=mpimg.imread('schematic.png')
print(schmn.shape)
schm=schmn[10:-80,190:1440,:]
imagebox1 = OffsetImage(schm, zoom=1)
ab1 = AnnotationBbox(imagebox1, (0.5,0.5),frameon=False,pad=0.1,xycoords='axes fraction')
ax00.add_artist(ab1)

ax00.set_xlim(0.2,0.8)
#ax00.set_ylim(0.4,0.6)
ax00.axis('off')

ax00.annotate(r'$(a)$',xy=(-0.65,1.48),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$(b)$',xy=(-0.65,0.44),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$(c)$',xy=(-0.65,-0.12),xycoords='axes fraction',zorder=np.inf)

ax00.annotate(r'$k\delta t$',xy=(0.55,1.48),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$(k+1)\delta t$',xy=(0.94,1.48),xycoords='axes fraction',zorder=np.inf)

ax00.annotate(r'Time',xy=(-0.3,1.48),xycoords='axes fraction',zorder=np.inf)
#ax00.arrow(0.2,0.,0.2,0,lw=0.1,color=(0.35,0.35,0.35),head_width=0.01,head_length=0.1,length_includes_head=True,clip_on=False,zorder=np.inf)
xa=-1.0
ya=0.6
ax00.annotate('', xy=(xa+0.9, ya+0.9),
             xycoords='axes fraction',
             xytext=(xa+1.06, ya+0.9),
             textcoords='axes fraction',
             arrowprops=dict(arrowstyle= '<|-',
                             color='k',
                             lw=3.5,
                             ls='-'),
             zorder=np.inf,
             clip_on=False
           )

ax00.annotate(r'$X_{i,[0,k]}$',xy=(-0.57,1.35),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$X_{l,[0,k]}$',xy=(-0.57,1.14),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$X_{j,[0,k]}$',xy=(-0.57,0.9),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$X_{j}(k+1)$',xy=(1.25,0.9),xycoords='axes fraction',zorder=np.inf)

ax00.annotate(r'$\Delta\mathcal{T}^{i\to j}_{[k,k+1]}$',xy=(0.76,1.33),xycoords='axes fraction',zorder=np.inf)

xs1=-0.42
ys1=-0.02

rounded_rect = FancyBboxPatch((0.99+xs1, 0.64+ys1), 0.145, 0.09, transform=ax00.transData,boxstyle='Round, pad=0,rounding_size=0.015', color='#fff2ccff', clip_on=False,zorder=np.inf)
ax00.add_patch(rounded_rect)
rounded_rect = FancyBboxPatch((1.165+xs1, 0.64+ys1), 0.14, 0.09, transform=ax00.transData,boxstyle='Round, pad=0,rounding_size=0.015', color='#f4ccccff', clip_on=False,zorder=np.inf)
ax00.add_patch(rounded_rect)
rounded_rect = FancyBboxPatch((0.75+xs1, 0.64+ys1), 0.22, 0.09, transform=ax00.transData,boxstyle='Round, pad=0,rounding_size=0.015', color='#d9ead3ff', clip_on=False,zorder=np.inf)
ax00.add_patch(rounded_rect)

ax00.annotate(r'$\Delta\mathcal{T}^{i\to j}_{[k,k+1]}=I\;(\; X_{j}(k+1)\; ;\;X_{i,[0,k]}\;| \; X_{j,[0,k]}\;)$',xy=(-0.27,0.65),xycoords='axes fraction',zorder=np.inf)

ax00.annotate(r'$X_{j,[0,N]}^{(\nu)}$',xy=(0.95,0.4),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$X_{i,[0,k]}^{(\mu)}$',xy=(0.31,0.2),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$X_{l,[0,k]}^{(\mu)}$',xy=(0.31,0.07),xycoords='axes fraction',zorder=np.inf)
#ax00.annotate(r'$P_{0}(X_{i,[0,k]}^{(\mu)},X_{l,[0,k]}^{(\mu)})$',xy=(0.31,0.17),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'Sample from',xy=(1.09,0.17),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'$P_{0}(X_{i,[0,k]},X_{l,[0,k]})$',xy=(0.93,0.07),xycoords='axes fraction',zorder=np.inf)

ax00.annotate(r'propagate dynamics',xy=(-0.35,-0.37),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'to sample from $P_{0}$',xy=(-0.33,-0.45),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'resample',xy=(0.68,-0.37),xycoords='axes fraction',zorder=np.inf)
ax00.annotate(r'with reweighting',xy=(0.55,-0.45),xycoords='axes fraction',zorder=np.inf)


plt.savefig('fig_schematic.png',bbox_inches='tight',pad_inches=0.1)

