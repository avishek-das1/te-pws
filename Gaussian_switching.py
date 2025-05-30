#Code to compute transfer entropy using the Gaussian approximation for model D with switching
from jpype import *
import random
import math
import numpy as np
import sys
from numba import jit,njit,prange

kxx=1.
kyy=1.
kzz=2.
kxy=-6.
kyx=-6.
kyz=0.
kzy=-kzz
kzx=0.
kxz=0.
Dx=1.
Dy=1.
Dz=1.
#timestep
dt=0.01

####force functions on x, y and z
@njit
def forcex(x,y,z):
    return -kxx*x-kxy*(1+x**2)/(1+x**2+y**2)-kxz*z

@njit
def forcey(x,y,z):
    return -kyy*y-kyx*(1+y**2)/(1+x**2+y**2)-kyz*z

@njit
def forcez(x,y,z):
    return -kzz*z-kzy*y-kzx*x

#function to propagate trajectories of x,y,z
@njit
def propagatexyz(xi,yi,zi,steps): #langevin equation to propagate x and y
    xt=np.zeros(steps+1+8000)
    yt=np.zeros(steps+1+8000)
    zt=np.zeros(steps+1+8000)
    xt[0]=xi
    yt[0]=yi
    zt[0]=zi

    for i in range(steps+8000):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fx=forcex(x,y,z)
        psi=np.random.normal()
        dx=fx*dt+(2*Dx*dt)**0.5*psi
        fy=forcey(x,y,z)
        psi2=np.random.normal()
        dy=fy*dt+(2*Dy*dt)**0.5*psi2
        fz=forcez(x,y,z)
        psi3=np.random.normal()
        dz=fz*dt+(2*Dz*dt)**0.5*psi3

        xt[i+1]=x+dx
        yt[i+1]=y+dy
        zt[i+1]=z+dz

    return xt[8000:],yt[8000:],zt[8000:]

numTrials=1 #number of trajectories
kHistoryLength=int(float(sys.argv[1])) #history length k+1
Nlen=int(float(sys.argv[2])) #number of timesteps in the trajectory
numObservations =  Nlen #int(1e7)

#function to calculate covariance matrix between two input trajectories at different values of delay times
@njit(parallel=True)
def carrcal(tr1,tr2):
    ltr=np.shape(tr1)[0]
    carrns=np.zeros((4,kHistoryLength+1)) #cxx,cxy,cyx,cyy
    countns=np.zeros(kHistoryLength+1) #counts for the average
    
    for i in prange(kHistoryLength+1):
        for j in range(ltr-i+1):
            carrns[0,i]+=tr1[j]*tr1[j+i]
            carrns[1,i]+=tr1[j]*tr2[j+i]
            carrns[2,i]+=tr2[j]*tr1[j+i]
            carrns[3,i]+=tr2[j]*tr2[j+i]
            countns[i]+=1

    return carrns,countns
 
carrs=np.zeros((4,kHistoryLength+1)) #cxx,cxy,cyx,cyy
counts=np.zeros(kHistoryLength+1) #counts for the average
ms=np.zeros(2)

#propagation of the trajectories
xn,_,zn=propagatexyz(0,6,6,numObservations)

#number of data sizes. The total size is divided by 3, by 2 or by 1, which are the elements of numlist.
numlist=np.array([3,2,1])
kh=kHistoryLength

for num in numlist:

    #build correlation functions
    carrs,counts=carrcal(xn[:int(numObservations/num)],zn[:int(numObservations/num)])
    ms[0]=np.average(xn[:int(numObservations/num)])
    ms[1]=np.average(zn[:int(numObservations/num)])

    carrs/=counts
    carrs[0,:]-=ms[0]*ms[0] #subtract means from second moments to get second cumulants
    carrs[1,:]-=ms[0]*ms[1]
    carrs[2,:]-=ms[1]*ms[0]
    carrs[3,:]-=ms[1]*ms[1]

    #Now build correlation matrices
    sinkss=carrs[3,0]
    sinksink_1k=np.zeros((1,kh))
    sinksink_kk=np.zeros((kh,kh))
    sinksinksource_12k=np.zeros((1,2*kh))
    sinksinksource_2k2k=np.zeros((2*kh,2*kh))
    
    #build submatrices from the full covariance matrix
    for i in range(kh): #sink first, source second; recent first, past second
        sinksink_1k[0,i]=carrs[3,i+1]
    
        sinksinksource_12k[0,i]=carrs[3,i+1]
        sinksinksource_12k[0,kh+i]=carrs[1,i+1]
    
        for j in range(i,kh):
            sinksink_kk[i,j]=carrs[3,j-i]
            sinksink_kk[j,i]=sinksink_kk[i,j]
    
            sinksinksource_2k2k[i,j]=carrs[3,j-i]
            sinksinksource_2k2k[j,i]=sinksinksource_2k2k[i,j]
            sinksinksource_2k2k[i,j+kh]=carrs[1,j-i]
            sinksinksource_2k2k[j+kh,i]=sinksinksource_2k2k[i,j+kh]
            sinksinksource_2k2k[i+kh,j]=carrs[2,j-i]
            sinksinksource_2k2k[j,i+kh]=sinksinksource_2k2k[i+kh,j]
            sinksinksource_2k2k[i+kh,j+kh]=carrs[0,j-i]
            sinksinksource_2k2k[j+kh,i+kh]=sinksinksource_2k2k[i+kh,j+kh]
    
    #Schur complement formula
    snum=sinkss-np.matmul(sinksink_1k,np.matmul(np.linalg.inv(sinksink_kk),np.transpose(sinksink_1k)))[0,0]
    sden=sinkss-np.matmul(sinksinksource_12k,np.matmul(np.linalg.inv(sinksinksource_2k2k),np.transpose(sinksinksource_12k)))[0,0]
    
    #return the transfer entropy (not divided by dt for rate yet)
    print(num,0.5*np.log(snum/sden))
    sys.stdout.flush()

