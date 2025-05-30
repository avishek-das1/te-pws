#code to use JIDT to compute the transfer entropy in model D using KSG
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

####force functions for x, y and z
@njit
def forcex(x,y,z):
    return -kxx*x-kxy*(1+x**2)/(1+x**2+y**2)-kxz*z

@njit
def forcey(x,y,z):
    return -kyy*y-kyx*(1+y**2)/(1+x**2+y**2)-kyz*z

@njit
def forcez(x,y,z):
    return -kzz*z-kzy*y-kzx*x

#propagating langevin trajectories
@njit
def propagatexyz(xi,yi,zi,steps):
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

# Change location of jar to match yours:
jarLocation = "infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation,"-Xmx20000M")

# Generate some random normalised data.
kHistoryLength=int(float(sys.argv[1])) #input the history length
Nlen=int(float(sys.argv[2])) #input the data size
numObservations =  Nlen #int(1e7)
xn,_,zn=propagatexyz(0,6,6,numObservations)
sourceArray=xn
destArray=zn

# Create a TE calculator and run it:
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
teCalc.setProperty("PROP_DYN_CORR_EXCL_TIME","6000") #theiler correction window

#number to divide the total data size into, for example, Nlen is divided by 3, by 2 or by 1 and so the elements of numlist are 3,2,1.
numlist=np.array([3,2,1])
ss=1 #downsampling rate
kh=kHistoryLength

for num in numlist:
    teCalc.initialise(kh,1,kh,1,1) # Use target history length of kHistoryLength (Schreiber k)
    teCalc.startAddObservations()
    teCalc.addObservations(JArray(JDouble, 1)(sourceArray[:int(numObservations/num):ss]), JArray(JDouble, 1)(destArray[:int(numObservations/num):ss])) #add the trajectory with the given downsampling rate
    teCalc.finaliseAddObservations()
    result = teCalc.computeAverageLocalOfObservations()
    print(kh,num,result) #output the result of the TE calculation. It must be divided by (ss*dt) to give a transfer entropy rate
    sys.stdout.flush()

