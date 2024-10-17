#Algorithm for Transfer Entropy- Path Weight Sampling (TE-PWS) in a 3-dimensional Chemical Reaction Network of species X,Y,Z of structure X <-> Y -> Z.
#Output is the Monte-Carlo estimate of MI and all transfer entropies
#Written by Avishek Das, Aug 30, 2024
#For other functional forms for the jump propensities, change all the different instances of the 'gillespie', 'propagate' and 'waitjumpprob' subroutines. Additionally, for conversion reactions (as opposed to catalysis), all other subroutines also need to be modified to account for simultaneous jumps in multiple trajectories.

import numpy as np
import sys
import math
from numba import jit,njit,prange

kxd=1. #rate constant for X death
kxn=1.#rate constant for X birth
kxy=0.1 #rate constant for X birth as catalyzed by Y
kyd=kxy+4 #rate constant for Y death
kyn=1. #rate constant for Y birth as catalyzed by X
kzd=50. #rate constant for Z death
kzn=50. #rate constant for Z birth as catalyzed by Y
ky0=0. #rate constant for Y birth from nothing

#every reaction must correspond to a unique stoichiometry change
Rtot=6 #total number of possible reactions
#Rtotxy is the number of reactions that change x and/or y, similar for Rtotxy etc...
Rtotxy=4 #not involved: kzn,kzd
Rtotyz=4 #not involved: kxn,kxd,kxy
Rtotxz=4 #not involved: kyn,ky0,kyd
Rtotx=2 #not involved: kyn,ky0,kyd,kzn,kzd
Rtoty=2 #not involved: kxn,kxd,kxy,kzn,kzd
Rtotz=2 #not involved: kxn,kxd,kxy,kyn,ky0,kyd

Rc=np.zeros((Rtot,3)) #array describing changes in copy numbers due to each reaction
Rc[0,:]=np.array([-1,0,0])
Rc[1,:]=np.array([+1,0,0])
Rc[2,:]=np.array([0,-1,0])
Rc[3,:]=np.array([0,+1,0])
Rc[4,:]=np.array([0,0,-1])
Rc[5,:]=np.array([0,0,+1])

Rcxy=np.zeros((Rtotxy,3))
Rcxy[0,:]=np.array([-1,0,0])
Rcxy[1,:]=np.array([+1,0,0])
Rcxy[2,:]=np.array([0,-1,0])
Rcxy[3,:]=np.array([0,+1,0])

Rcyz=np.zeros((Rtotyz,3))
Rcyz[0,:]=np.array([0,-1,0])
Rcyz[1,:]=np.array([0,+1,0])
Rcyz[2,:]=np.array([0,0,-1])
Rcyz[3,:]=np.array([0,0,+1])

Rcxz=np.zeros((Rtotxz,3))
Rcxz[0,:]=np.array([-1,0,0])
Rcxz[1,:]=np.array([+1,0,0])
Rcxz[2,:]=np.array([0,0,-1])
Rcxz[3,:]=np.array([0,0,+1])

Rcx=np.zeros((Rtotx,3))
Rcx[0,:]=np.array([-1,0,0])
Rcx[1,:]=np.array([+1,0,0])

Rcy=np.zeros((Rtoty,3))
Rcy[0,:]=np.array([0,-1,0])
Rcy[1,:]=np.array([0,+1,0])

Rcz=np.zeros((Rtotz,3))
Rcz[0,:]=np.array([0,0,-1])
Rcz[1,:]=np.array([0,0,+1])

#timestep
dt=0.005 #for RR scheme resampling and for computing TE estimate J+E
#when increasing dt, always increase Mmin

stepmags=np.arange(1,2001,1) #number of steps
ttot=stepmags[-1]*dt #total trajectory duration

N=160 #=M_{1} in manuscript=number of trajectories for the outer Monte-Carlo average
K=1000 #=M_{2} in manuscript=number of trajectories for marginalization
Mmin=100 #maximum allowed number of reactions within dt time, if this is exceeded during propagation of the dynamics, segmentation fault occurs due to array overflow. Mmin must be increased when increasing dt. This results in higher memory load.

lowprob=np.exp(-500) #exp(loglowprob) is what probability 0 is replaced by, for bookkeeping purposes

tol=1e-5 #tolerance for initial condition histogram, below which the probabilities are put to zero
tolm=1e-8 #machine precision tolerance for comparing floats
indend=-int(1e7) #dummy array index that should never be used in subsequent calculations
infwaittime=1e7 #proxy for a really long wait time that should never be used in subsequent calculations

@njit 
def updatewait(waitsfull,indn,waitcarry,statesn,wait0): 
#for a given waiting time for a reaction already determined (wait0), determine how much should the pointer (indn,waitcarry) to the waiting time trajectory (waitsfull) be slid forward to make it in sync 
#waitsfull is the full waiting times trajectory, indn is the pointer to the ongoing waiting time period, waitcarry is the remaining time left in the current period, statesn is the total number of waiting times (i.e., total number of reactions plus one) in the trajectory, wait0 is the time by which to slide the clock forward
    segmentwait=waitcarry
    for i in range(indn,statesn):
        if i==indn:
            waitold=0.
        else:
            waitold+=segmentwait
            
        if i>indn:
            segmentwait=waitsfull[i]
        
        if wait0<=waitold+segmentwait: #this is the new segment where the pointer should point
            indn=i
            waitcarry=waitold+segmentwait-wait0
            break
        elif i==statesn-1: #reached the end of the trajectory without new reaction
            indn=indend #number of states has reached the endpoint, so this index should never be accessed
            waitcarry=0
            
    return indn,waitcarry

@njit
def mergewaitsxy(xt,yt,zt,waitsfull,indn,waitcarry,statesn,newwaits): 
#The mergewaits subroutines merge (in a sorted fashion) two waiting time lists in two given subspaces.
#mergewaitsxy merges 'newwaits' in the space (x,y) with 'waitsfull,indn,waitcarry,statesn' in the space (z).
#For explanation of waitsfull, indn, waitcarry and statesn, see the 'updatewait' subroutine
#IMPORTANT: defined only for a time interval dt, cannot merge longer waiting time trajectories
    t=0.
    tfull=dt
    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    mergedwaits=np.zeros(Mp+1)
    xtn=np.zeros(Mp+1)
    ytn=np.zeros(Mp+1)
    ztn=np.zeros(Mp+1)
    i=0
    iA=0
    iB=indn
    tflag=0 #flag for when t passes tfull
    
    segmentwaitA=newwaits[iA]
    segmentwaitB=waitcarry

    segmentdiff=segmentwaitA-segmentwaitB
    while tflag==0:
        while segmentdiff>0: #B occurs earlier than A in the current segments
            xtn[i]=xt[iA]
            ytn[i]=yt[iA]
            ztn[i]=zt[iB]
            mergedwaits[i]=segmentwaitB
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iB+=1
            segmentwaitA-=segmentwaitB
            segmentwaitB=waitsfull[iB]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB 
        
        while segmentdiff<0: #A occurs earlier than B in the current segments
            xtn[i]=xt[iA]
            ytn[i]=yt[iA]
            ztn[i]=zt[iB]
            mergedwaits[i]=segmentwaitA
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iA+=1
            segmentwaitB-=segmentwaitA
            segmentwaitA=newwaits[iA]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB
    
    return xtn,ytn,ztn, mergedwaits,i+1 #returns x,y,z trajectories sorted according to the temporal sequence of the merged waits list. i+1 is the total number of states in the merged list within duration dt.

@njit
def mergewaitsyz(xt,yt,zt,waitsfull,indn,waitcarry,statesn,newwaits): 
#for explanation see subroutine 'mergewaitsxy'
    t=0.
    tfull=dt
    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    mergedwaits=np.zeros(Mp+1)
    xtn=np.zeros(Mp+1)
    ytn=np.zeros(Mp+1)
    ztn=np.zeros(Mp+1)
    i=0
    iA=0
    iB=indn
    tflag=0 #flag for when t passes tfull
    
    segmentwaitA=newwaits[iA]
    segmentwaitB=waitcarry

    segmentdiff=segmentwaitA-segmentwaitB
    while tflag==0:
        while segmentdiff>0: #B occurs earlier
            xtn[i]=xt[iB]
            ytn[i]=yt[iA]
            ztn[i]=zt[iA]
            mergedwaits[i]=segmentwaitB
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iB+=1
            segmentwaitA-=segmentwaitB
            segmentwaitB=waitsfull[iB]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB 
        
        while segmentdiff<0: #A occurs earlier
            xtn[i]=xt[iB]
            ytn[i]=yt[iA]
            ztn[i]=zt[iA]
            mergedwaits[i]=segmentwaitA
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iA+=1
            segmentwaitB-=segmentwaitA
            segmentwaitA=newwaits[iA]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB
    
    return xtn,ytn,ztn, mergedwaits,i+1

@njit
def mergewaitsxz(xt,yt,zt,waitsfull,indn,waitcarry,statesn,newwaits): 
#for explanation see subroutine 'mergewaitsxy'
    t=0.
    tfull=dt
    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    mergedwaits=np.zeros(Mp+1)
    xtn=np.zeros(Mp+1)
    ytn=np.zeros(Mp+1)
    ztn=np.zeros(Mp+1)
    i=0
    iA=0
    iB=indn
    tflag=0 #flag for when t passes tfull
    
    segmentwaitA=newwaits[iA]
    segmentwaitB=waitcarry

    segmentdiff=segmentwaitA-segmentwaitB
    while tflag==0:
        while segmentdiff>0: #B occurs earlier
            xtn[i]=xt[iA]
            ytn[i]=yt[iB]
            ztn[i]=zt[iA]
            mergedwaits[i]=segmentwaitB
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iB+=1
            segmentwaitA-=segmentwaitB
            segmentwaitB=waitsfull[iB]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB 
        
        while segmentdiff<0: #A occurs earlier
            xtn[i]=xt[iA]
            ytn[i]=yt[iB]
            ztn[i]=zt[iA]
            mergedwaits[i]=segmentwaitA
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iA+=1
            segmentwaitB-=segmentwaitA
            segmentwaitA=newwaits[iA]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB
    
    return xtn,ytn,ztn, mergedwaits,i+1

@njit
def mergewaitsx(xt,yt,zt,waitsfull,indn,waitcarry,statesn,newwaits): 
#for explanation see subroutine 'mergewaitsxy'
    t=0.
    tfull=dt
    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    mergedwaits=np.zeros(Mp+1)
    xtn=np.zeros(Mp+1)
    ytn=np.zeros(Mp+1)
    ztn=np.zeros(Mp+1)
    i=0
    iA=0
    iB=indn
    tflag=0 #flag for when t passes tfull
    
    segmentwaitA=newwaits[iA]
    segmentwaitB=waitcarry

    segmentdiff=segmentwaitA-segmentwaitB
    while tflag==0:
        while segmentdiff>0: #B occurs earlier
            xtn[i]=xt[iA]
            ytn[i]=yt[iB]
            ztn[i]=zt[iB]
            mergedwaits[i]=segmentwaitB
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iB+=1
            segmentwaitA-=segmentwaitB
            segmentwaitB=waitsfull[iB]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB 
        
        while segmentdiff<0: #A occurs earlier
            xtn[i]=xt[iA]
            ytn[i]=yt[iB]
            ztn[i]=zt[iB]
            mergedwaits[i]=segmentwaitA
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iA+=1
            segmentwaitB-=segmentwaitA
            segmentwaitA=newwaits[iA]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB
    
    return xtn,ytn,ztn, mergedwaits,i+1

@njit
def mergewaitsy(xt,yt,zt,waitsfull,indn,waitcarry,statesn,newwaits): 
#for explanation see subroutine 'mergewaitsxy'
    t=0.
    tfull=dt
    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    mergedwaits=np.zeros(Mp+1)
    xtn=np.zeros(Mp+1)
    ytn=np.zeros(Mp+1)
    ztn=np.zeros(Mp+1)
    i=0
    iA=0
    iB=indn
    tflag=0 #flag for when t passes tfull
    
    segmentwaitA=newwaits[iA]
    segmentwaitB=waitcarry

    segmentdiff=segmentwaitA-segmentwaitB
    while tflag==0:
        while segmentdiff>0: #B occurs earlier
            xtn[i]=xt[iB]
            ytn[i]=yt[iA]
            ztn[i]=zt[iB]
            mergedwaits[i]=segmentwaitB
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iB+=1
            segmentwaitA-=segmentwaitB
            segmentwaitB=waitsfull[iB]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB 
        
        while segmentdiff<0: #A occurs earlier
            xtn[i]=xt[iB]
            ytn[i]=yt[iA]
            ztn[i]=zt[iB]
            mergedwaits[i]=segmentwaitA
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iA+=1
            segmentwaitB-=segmentwaitA
            segmentwaitA=newwaits[iA]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB
    
    return xtn,ytn,ztn, mergedwaits,i+1

@njit
def mergewaitsz(xt,yt,zt,waitsfull,indn,waitcarry,statesn,newwaits): 
#for explanation see subroutine 'mergewaitsxy'
    t=0.
    tfull=dt
    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    mergedwaits=np.zeros(Mp+1)
    xtn=np.zeros(Mp+1)
    ytn=np.zeros(Mp+1)
    ztn=np.zeros(Mp+1)
    i=0
    iA=0
    iB=indn
    tflag=0 #flag for when t passes tfull
    
    segmentwaitA=newwaits[iA]
    segmentwaitB=waitcarry

    segmentdiff=segmentwaitA-segmentwaitB
    while tflag==0:
        while segmentdiff>0: #B occurs earlier
            xtn[i]=xt[iB]
            ytn[i]=yt[iB]
            ztn[i]=zt[iA]
            mergedwaits[i]=segmentwaitB
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iB+=1
            segmentwaitA-=segmentwaitB
            segmentwaitB=waitsfull[iB]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB 
        
        while segmentdiff<0: #A occurs earlier
            xtn[i]=xt[iB]
            ytn[i]=yt[iB]
            ztn[i]=zt[iA]
            mergedwaits[i]=segmentwaitA
            t+=mergedwaits[i]
            if t>tfull:
                tflag=1
                break
            iA+=1
            segmentwaitB-=segmentwaitA
            segmentwaitA=newwaits[iA]
            i+=1
            segmentdiff=segmentwaitA-segmentwaitB
    
    return xtn,ytn,ztn, mergedwaits,i+1


@njit
def gillespiexyz(x,y,z,psi,psi2): 
#The gillespie subroutines return the waiting time and the identity of the next reaction, within a given space, in this case (x,y,z)
    Rp=np.zeros(Rtot) #cumulative reaction propensities

    Rp[0]=kxd*x
    Rp[1]=Rp[0]+kxn+kxy*y

    Rp[2]=Rp[1]+kyd*y
    Rp[3]=Rp[2]+kyn*x+ky0

    Rp[4]=Rp[3]+kzd*z
    Rp[5]=Rp[4]+kzn*y

    Qval=-Rp[-1] #escape rate
    Rp/=Rp[-1] #normalized reaction probabilities in the different channels
    if Qval<0:
        waitn=np.log(psi)/Qval
    else:
        waitn=infwaittime #if there are no reactions, Qval is zero by construction, in which case the waiting time should be infinity
    Rin=np.searchsorted(Rp,psi2) #reaction index

    return waitn,Rin

@njit
def gillespiexy(x,y,zt,waitsfull,indn,waitcarry,statesn,psi,psi2): 
#returns waiting time and the identity of the next reaction in the (x,y) space, given a frozen zt trajectory, the latter being accompanied by (waitsfull,indn,waitcarry,statesn)
#psi and psi2 are the two uniform random numbers to be used to determine the reaction time and the reaction identity
#for explanations about (waitsfull,indn,waitcarry,statesn), see subroutine 'updatewait'
#for explanations about the variables in this subroutine, see subroutine 'gillespiexyz'

    Rp=np.zeros(Rtotxy)

    Rp[0]=kxd*x
    Rp[1]=Rp[0]+kxn+kxy*y

    Rp[2]=Rp[1]+kyd*y
    Rp[3]=Rp[2]+kyn*x+ky0

    Qval=-Rp[-1]
    Rp/=Rp[-1]
    if Qval<0:
        waitn=np.log(psi)/Qval
    else:
        waitn=infwaittime
    Rxyin=np.searchsorted(Rp,psi2) #reaction index

    #a dummy search through waiting times index to figure out the positions of the new pointers for the zt trajectory
    indn,waitcarry=updatewait(waitsfull,indn,waitcarry,statesn,waitn)

    return waitn,Rxyin,indn,waitcarry

@njit
def gillespieyz(xt,y,z,waitsfull,indn,waitcarry,statesn,psi,psi2):
#for more explanations see subroutines 'gillespiexyz' and 'gillespiexy'

    Rp=np.zeros(Rtotyz)

    Rp[0]=kyd*y
    Rp[1]=Rp[0]+ky0 #the time-dependent part of the propensity of the birth of y given a frozen xt trajectory is dealt with below

    Rp[2]=Rp[1]+kzd*z
    Rp[3]=Rp[2]+kzn*y

    #This loop is for time-dependent propensities. Loop through the given xt to find the new waiting time for y.
    segmentwait=waitcarry
    for i in range(indn,statesn):
        if i==indn:
            Qtimestold=0.
            waitold=0.
            Rp[1:]+=kyn*xt[i]
            Qval=-Rp[-1]
            if Qval<0:
                waitn=np.log(psi)/Qval #hypothetical waiting time according to the current escape propensity
            else:
                waitn=infwaittime
        else:
            Qtimestold+=Qval*segmentwait
            waitold+=segmentwait
            Rp[1:]+=kyn*(-xt[i-1]+xt[i])
            Qval=-Rp[-1]
            if Qval<0:
                waitn=(np.log(psi)-Qtimestold)/Qval #hypothetical waiting time according to the cumulative escape propensity from all segments till a given future segments
            else:
                waitn=infwaittime
            
        if i>indn:
            segmentwait=waitsfull[i]
        
        if waitn<segmentwait: #waiting time is less than the cumulative time of all the segments, hence this is the correct waiting time
            indn=i
            waitcarry=segmentwait-waitn
            waitn+=waitold
            break
        elif i==statesn-1: #reached the end of the trajectory without new reaction
            indn=indend #number of states has reached the endpoint, so this index should never be accessed
            waitcarry=0
            waitn+=waitold

    Rp/=Rp[-1]
    Ryzin=np.searchsorted(Rp,psi2) #reaction index

    return waitn,Ryzin,indn,waitcarry 

@njit
def gillespiexz(x,yt,z,waitsfull,indn,waitcarry,statesn,psi,psi2):
#for more explanations see subroutines 'gillespiexyz', 'gillespiexy' and 'gillespieyz'

    Rp=np.zeros(Rtotxz)

    Rp[0]=kxd*x
    Rp[1]=Rp[0]+kxn #deal with time-dependent propensity a few code lines below

    Rp[2]=Rp[1]+kzd*z
    Rp[3]=Rp[2] #deal with time-dependent propensity a few code lines below

    #this loop is for time-dependent propensities
    segmentwait=waitcarry
    for i in range(indn,statesn):
        if i==indn:
            Qtimestold=0.
            waitold=0.
            Rp[1:]+=kxy*yt[i]
            Rp[3:]+=kzn*yt[i]
            Qval=-Rp[-1]
            if Qval<0:
                waitn=np.log(psi)/Qval
            else:
                waitn=infwaittime
        else:
            Qtimestold+=Qval*segmentwait
            waitold+=segmentwait
            Rp[1:]+=kxy*(-yt[i-1]+yt[i])
            Rp[3:]+=kzn*(-yt[i-1]+yt[i])
            Qval=-Rp[-1]
            if Qval<0:
                waitn=(np.log(psi)-Qtimestold)/Qval
            else:
                waitn=infwaittime
            
        if i>indn:
            segmentwait=waitsfull[i]
        
        if waitn<segmentwait: #new reaction time determined
            indn=i
            waitcarry=segmentwait-waitn
            waitn+=waitold
            break
        elif i==statesn-1: #reached the end of the trajectory without new reaction
            indn=indend #number of states has reached the endpoint, so this index should never be accessed
            waitcarry=0
            waitn+=waitold

    Rp/=Rp[-1]
    Rxzin=np.searchsorted(Rp,psi2) #reaction index

    return waitn,Rxzin,indn,waitcarry 

@njit
def gillespiex(x,yt,zt,waitsfull,indn,waitcarry,statesn,psi,psi2): 
#for more explanations see subroutines 'gillespiexyz', 'gillespiexy' and 'gillespieyz'

    Rp=np.zeros(Rtotx)

    Rp[0]=kxd*x
    Rp[1]=Rp[0]+kxn #deal with time-dependent propensity a few code lines below

    #this loop is for time-dependent propensities
    segmentwait=waitcarry
    for i in range(indn,statesn):
        if i==indn:
            Qtimestold=0.
            waitold=0.
            Rp[1:]+=kxy*yt[i]
            Qval=-Rp[-1]
            if Qval<0:
                waitn=np.log(psi)/Qval
            else:
                waitn=infwaittime
        else:
            Qtimestold+=Qval*segmentwait
            waitold+=segmentwait
            Rp[1:]+=kxy*(-yt[i-1]+yt[i])
            Qval=-Rp[-1]
            if Qval<0:
                waitn=(np.log(psi)-Qtimestold)/Qval
            else:
                waitn=infwaittime
            
        if i>indn:
            segmentwait=waitsfull[i]
        
        if waitn<segmentwait: #new reaction time determined
            indn=i
            waitcarry=segmentwait-waitn
            waitn+=waitold
            break
        elif i==statesn-1: #reached the end of the trajectory without new reaction
            indn=indend #number of states has reached the endpoint, so this index should never be accessed
            waitcarry=0
            waitn+=waitold

    Rp/=Rp[-1]
    Rxin=np.searchsorted(Rp,psi2) #reaction index

    return waitn,Rxin,indn,waitcarry 

@njit
def gillespiey(xt,y,zt,waitsfull,indn,waitcarry,statesn,psi,psi2):
#for more explanations see subroutines 'gillespiexyz', 'gillespiexy' and 'gillespieyz'

    Rp=np.zeros(Rtoty)

    Rp[0]=kyd*y
    Rp[1]=Rp[0]+ky0 #deal with time-dependent propensity a few code lines below

    #this loop is for time-dependent propensities
    segmentwait=waitcarry
    for i in range(indn,statesn):
        if i==indn:
            Qtimestold=0.
            waitold=0.
            Rp[1:]+=kyn*xt[i]
            Qval=-Rp[-1]
            if Qval<0:
                waitn=np.log(psi)/Qval
            else:
                waitn=infwaittime
        else:
            Qtimestold+=Qval*segmentwait
            waitold+=segmentwait
            Rp[1:]+=kyn*(-xt[i-1]+xt[i])
            Qval=-Rp[-1]
            if Qval<0:
                waitn=(np.log(psi)-Qtimestold)/Qval
            else:
                waitn=infwaittime
            
        if i>indn:
            segmentwait=waitsfull[i]
        
        if waitn<segmentwait: #new reaction time determined
            indn=i
            waitcarry=segmentwait-waitn
            waitn+=waitold
            break
        elif i==statesn-1: #reached the end of the trajectory without new reaction
            indn=indend #number of states has reached the endpoint, so this index should never be accessed
            waitcarry=0
            waitn+=waitold

    Rp/=Rp[-1]
    Ryin=np.searchsorted(Rp,psi2) #reaction index

    return waitn,Ryin,indn,waitcarry


@njit
def gillespiez(xt,yt,z,waitsfull,indn,waitcarry,statesn,psi,psi2):
#for more explanations see subroutines 'gillespiexyz', 'gillespiexy' and 'gillespieyz'

    Rp=np.zeros(Rtotz) 

    Rp[0]=kzd*z
    Rp[1]=Rp[0] #deal with time-dependent propensity a few code lines below

    #this loop is for time-dependent propensities
    segmentwait=waitcarry
    for i in range(indn,statesn):
        if i==indn:
            Qtimestold=0.
            waitold=0.
            Rp[1:]+=kzn*yt[i]
            Qval=-Rp[-1]
            if Qval<0:
                waitn=np.log(psi)/Qval
            else:
                waitn=infwaittime
        else:
            Qtimestold+=Qval*segmentwait
            waitold+=segmentwait
            Rp[1:]+=kzn*(-yt[i-1]+yt[i])
            Qval=-Rp[-1]
            if Qval<0:
                waitn=(np.log(psi)-Qtimestold)/Qval
            else:
                waitn=infwaittime
            
        if i>indn:
            segmentwait=waitsfull[i]
        
        if waitn<segmentwait: #new reaction time determined
            indn=i
            waitcarry=segmentwait-waitn
            waitn+=waitold
            break
        elif i==statesn-1: #reached the end of the trajectory without new reaction
            indn=indend #number of states has reached the endpoint, so this index should never be accessed
            waitcarry=0
            waitn+=waitold

    Rp/=Rp[-1]
    Rzin=np.searchsorted(Rp,psi2) #reaction index

    return waitn,Rzin,indn,waitcarry

@njit
def propagatexyz(xi,yi,zi,wait0,Rin0,fflag,steps): 
#Gillespie algorithm to propagate x,y,z for time period steps*dt; wait0 is waiting time left in the current segment, Rin0 is next reaction index
#fflag is flag for whether the first wait and reaction is already fixed or should be sampled again.

    Mp=steps*Mmin
    xt=np.zeros(Mp+1) #copy numbers
    yt=np.zeros(Mp+1)
    zt=np.zeros(Mp+1)
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    xt[0]=xi
    yt[0]=yi
    zt[0]=zi
    
    #zeroth step, if fflag==1, then use the given fixed wait0 and Rin0.
    if fflag==1:
        x=xt[ind]
        y=yt[ind]
        z=zt[ind]
        waits[ind]=wait0
        Rin=Rin0
        xt[ind+1]=x+Rc[Rin0,0]
        yt[ind+1]=y+Rc[Rin0,1]
        zt[ind+1]=z+Rc[Rin0,2]
        t+=wait0
        ind+=1

    while t<tfull:
        x=xt[ind]
        y=yt[ind]
        z=zt[ind]

        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin=gillespiexyz(x,y,z,psi,psi2)
        xt[ind+1]=x+Rc[Rin,0]
        yt[ind+1]=y+Rc[Rin,1]
        zt[ind+1]=z+Rc[Rin,2]

        t+=waits[ind]
        ind+=1

    return xt,yt,zt,waits,ind
    
@njit
def propagatexy(xi,yi,zt,wait0,Rinxy0,waitsfull,indn3,waitcarry3,statesn,fflag,steps): 
#Gillespie algorithm to propagate x,y for time period steps*dt
#Given that the frozen zt trajectory has accompanying waiting time list (waitsfull,indn3,waitcarry3,statesn)
#for more explanations see subroutine 'propagatexyz'

    Mp=np.amax(np.array([Mmin,steps])) #size of holder for reactions
    xt=np.zeros(Mp+1) #copy numbers
    yt=np.zeros(Mp+1)
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    xt[0]=xi
    yt[0]=yi

    #zeroth step
    if fflag==1:
        x=xt[ind]
        y=yt[ind]
        indn=indn3
        waitcarry=waitcarry3
        waits[ind]=wait0
        Rin=Rinxy0
        xt[ind+1]=x+Rcxy[Rinxy0,0]
        yt[ind+1]=y+Rcxy[Rinxy0,1]
        t+=wait0
        ind+=1
        indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,wait0)

    while t<tfull:
        x=xt[ind]
        y=yt[ind]
        indn=indn3
        waitcarry=waitcarry3
        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin,indn3,waitcarry3=gillespiexy(x,y,zt,waitsfull,indn,waitcarry,statesn,psi,psi2) 
        xt[ind+1]=x+Rcxy[Rin,0]
        yt[ind+1]=y+Rcxy[Rin,1]

        t+=waits[ind]
        ind+=1
    
    wait0n=t-tfull #carryover waiting time for the next segment after the end of this trajectory
    indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,waits[ind-1]-wait0n) #What does waits[ind-1]-wait0n mean? That's the last segment that is within the trajectory. This line adjusts the pointer to the zt waiting time list such that it does not include the time in the last segment. This is needed separately from the 'gillespiexy' call, because indn3 and waitcarry3 does not have the updated pointers after the last 'while' loop, as indn3 and waitcarry3 are only updated at the beginning of the while loop.
    
    return xt,yt,waits,ind,wait0n,Rin,indn3,waitcarry3 #waitcarry3 is for zt, wait0n is for xy

@njit
def propagateyz(xt,yi,zi,wait0,Rinyz0,waitsfull,indn3,waitcarry3,statesn,fflag,steps): 
#Gillespie algorithm to propagate y,z for time period steps*dt
#Given that the frozen xt trajectory has accompanying waiting time list (waitsfull,indn3,waitcarry3,statesn)
#for more explanations see subroutines 'propagatexyz' and 'propagatexy'
    Mp=np.amax(np.array([Mmin,steps])) #size of holder for reactions
    yt=np.zeros(Mp+1) #copy numbers
    zt=np.zeros(Mp+1)
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    yt[0]=yi
    zt[0]=zi

    #zeroth step
    if fflag==1:
        y=yt[ind]
        z=zt[ind]
        indn=indn3
        waitcarry=waitcarry3
        waits[ind]=wait0
        Rin=Rinyz0
        yt[ind+1]=y+Rcyz[Rinyz0,1]
        zt[ind+1]=z+Rcyz[Rinyz0,2]
        t+=wait0
        ind+=1
        indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,wait0)

    while t<tfull:
        y=yt[ind]
        z=zt[ind]
        indn=indn3
        waitcarry=waitcarry3
        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin,indn3,waitcarry3=gillespieyz(xt,y,z,waitsfull,indn,waitcarry,statesn,psi,psi2) 
        yt[ind+1]=y+Rcyz[Rin,1]
        zt[ind+1]=z+Rcyz[Rin,2]

        t+=waits[ind]
        ind+=1

    #ind-=1
    wait0n=t-tfull
    indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,waits[ind-1]-wait0n)
    return yt,zt,waits,ind,wait0n,Rin,indn3,waitcarry3

@njit
def propagatexz(xi,yt,zi,wait0,Rinxz0,waitsfull,indn3,waitcarry3,statesn,fflag,steps): 
#Gillespie algorithm to propagate x,z for time period steps*dt
#Given that the frozen yt trajectory has accompanying waiting time list (waitsfull,indn3,waitcarry3,statesn)
#for more explanations see subroutines 'propagatexyz' and 'propagatexy'
    Mp=np.amax(np.array([Mmin,steps])) #size of holder for reactions
    xt=np.zeros(Mp+1) #copy numbers
    zt=np.zeros(Mp+1)
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    xt[0]=xi
    zt[0]=zi

    #zeroth step
    if fflag==1:
        x=xt[ind]
        z=zt[ind]
        indn=indn3
        waitcarry=waitcarry3
        waits[ind]=wait0
        Rin=Rinxz0
        xt[ind+1]=x+Rcxz[Rinxz0,0]
        zt[ind+1]=z+Rcxz[Rinxz0,2]
        t+=wait0
        ind+=1
        indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,wait0)

    while t<tfull:
        x=xt[ind]
        z=zt[ind]
        indn=indn3
        waitcarry=waitcarry3
        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin,indn3,waitcarry3=gillespiexz(x,yt,z,waitsfull,indn,waitcarry,statesn,psi,psi2)
        xt[ind+1]=x+Rcxz[Rin,0]
        zt[ind+1]=z+Rcxz[Rin,2]

        t+=waits[ind]
        ind+=1

    #ind-=1
    wait0n=t-tfull
    indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,waits[ind-1]-wait0n)
    return xt,zt,waits,ind,wait0n,Rin,indn3,waitcarry3

@njit
def propagatex(xi,yt,zt,wait0,Rinx0,waitsfull,indn3,waitcarry3,statesn,fflag,steps):
#Gillespie algorithm to propagate x for time period steps*dt
#Given that the frozen yt,zt trajectories have an accompanying waiting time list (waitsfull,indn3,waitcarry3,statesn)
#for more explanations see subroutines 'propagatexyz' and 'propagatexy'

    Mp=np.amax(np.array([Mmin,steps])) #size of holder for reactions
    xt=np.zeros(Mp+1) #copy numbers
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    xt[0]=xi

    #zeroth step
    if fflag==1:
        x=xt[ind]
        indn=indn3
        waitcarry=waitcarry3
        waits[ind]=wait0
        Rin=Rinx0
        xt[ind+1]=x+Rcx[Rinx0,0]
        t+=wait0
        ind+=1
        indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,wait0)
    
    while t<tfull:
        x=xt[ind]
        indn=indn3
        waitcarry=waitcarry3
        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin,indn3,waitcarry3=gillespiex(x,yt,zt,waitsfull,indn,waitcarry,statesn,psi,psi2)
        xt[ind+1]=x+Rcx[Rin,0]

        t+=waits[ind]
        ind+=1

    #ind-=1
    wait0n=t-tfull
    indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,waits[ind-1]-wait0n)
    return xt,waits,ind,wait0n,Rin,indn3,waitcarry3

@njit
def propagatey(xt,yi,zt,wait0,Riny0,waitsfull,indn3,waitcarry3,statesn,fflag,steps):
#Gillespie algorithm to propagate y for time period steps*dt
#Given that the frozen xt,zt trajectories have an accompanying waiting time list (waitsfull,indn3,waitcarry3,statesn)
#for more explanations see subroutines 'propagatexyz' and 'propagatexy'

    Mp=np.amax(np.array([Mmin,steps])) #size of holder for reactions
    yt=np.zeros(Mp+1) #copy numbers
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    yt[0]=yi

    #zeroth step
    if fflag==1:
        y=yt[ind]
        indn=indn3
        waitcarry=waitcarry3
        waits[ind]=wait0
        Rin=Riny0
        yt[ind+1]=y+Rcy[Riny0,1]
        t+=wait0
        ind+=1
        indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,wait0)

    while t<tfull:
        y=yt[ind]
        indn=indn3
        waitcarry=waitcarry3
        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin,indn3,waitcarry3=gillespiey(xt,y,zt,waitsfull,indn,waitcarry,statesn,psi,psi2)
        yt[ind+1]=y+Rcy[Rin,1]

        t+=waits[ind]
        ind+=1

    #ind-=1
    wait0n=t-tfull
    indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,waits[ind-1]-wait0n)
    return yt,waits,ind,wait0n,Rin,indn3,waitcarry3

@njit
def propagatez(xt,yt,zi,wait0,Rinz0,waitsfull,indn3,waitcarry3,statesn,fflag,steps): 
#Gillespie algorithm to propagate z for time period steps*dt
#Given that the frozen xt,yt trajectories have an accompanying waiting time list (waitsfull,indn3,waitcarry3,statesn)
#for more explanations see subroutines 'propagatexyz' and 'propagatexy'

    Mp=np.amax(np.array([Mmin,steps])) #size of holder for reactions
    zt=np.zeros(Mp+1) #copy numbers
    waits=np.zeros(Mp+1) #waiting times for each reaction
    ind=0 #current index in the trajectory array
    t=0.
    tfull=steps*dt #total time needed

    zt[0]=zi

    #zeroth step
    if fflag==1:
        z=zt[ind]
        indn=indn3
        waitcarry=waitcarry3
        waits[ind]=wait0
        Rin=Rinz0
        zt[ind+1]=z+Rcz[Rinz0,2]
        t+=wait0
        ind+=1
        indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,wait0)

    while t<tfull:
        z=zt[ind]
        indn=indn3
        waitcarry=waitcarry3
        psi=np.random.uniform() #for waiting time
        psi2=np.random.uniform() #for jump choice
        waits[ind],Rin,indn3,waitcarry3=gillespiez(xt,yt,z,waitsfull,indn,waitcarry,statesn,psi,psi2)
        zt[ind+1]=z+Rcz[Rin,2]

        t+=waits[ind]
        ind+=1

    #ind-=1
    wait0n=t-tfull
    indn3,waitcarry3=updatewait(waitsfull,indn,waitcarry,statesn,waits[ind-1]-wait0n)
    return zt,waits,ind,wait0n,Rin,indn3,waitcarry3
    
@njit
def waitjumpprobxyz(x,y,z,wait,xn,yn,zn):
#Given old (x,y,z) before jump, new (x,y,z) after jump and waiting time till the jump, this subroutine returns the log of the waiting probability in the full (x,y,z) space, individual waiting probabilities in (x,y,z), the identity of the space in which the jump occurs, and the log of the jump probability.
    Rp=np.zeros(Rtot) 
    
    #calculate propensities
    Rp[0]=kxd*x
    Rp[1]=Rp[0]+kxn+kxy*y

    Rp[2]=Rp[1]+kyd*y
    Rp[3]=Rp[2]+kyn*x+ky0

    Rp[4]=Rp[3]+kzd*z
    Rp[5]=Rp[4]+kzn*y

    Qval=-Rp[-1]

    waitprob=Qval*wait #log of waiting time probability

    #For conversion reactions, the jump probability sections need to be modified

    #search through all reactions to find which reaction fired
    delx=xn-x
    dely=yn-y
    delz=zn-z
    jumpprob=1.
    jumpid=-1
    for i in range(Rtot):
        if np.abs(Rc[i,0]-delx)<tolm and np.abs(Rc[i,1]-dely)<tolm and np.abs(Rc[i,2]-delz)<tolm: #This reaction occurred
            if i==0:
                jumpprob=np.amax(np.array([lowprob,Rp[i]]))
            else:
                jumpprob=np.amax(np.array([lowprob,Rp[i]-Rp[i-1]]))
            if i==0 or i==1: #x was changed
                jumpid=0
            if i==2 or i==3: #y was changed
                jumpid=1
            if i==4 or i==5: #z was changed #This assumed only one species changes at a time, make other categories if that is not the case
                jumpid=2
            break

    return waitprob, np.log(jumpprob),jumpid,-Rp[1]*wait,-(Rp[3]-Rp[1])*wait,-(Rp[5]-Rp[3])*wait #last three are waitprob within reduced spaces of x,y and z.

@njit
def trajprob(xt,yt,zt,waits,ind): #it gives probabilities of the trajectory within the full 3D space as well as within reduced spaces. 
#written by construction only for a duration dt

    p=0. #holder for trajectory probability for the full trajectory in the full 3D space

    t=0.
    tfull=dt #total time
    problist=np.zeros((3,Mmin)) #stores trajectory probability in full space, upto different jump times (cumulatively) in x,y,z
    problist0=np.zeros((3,3,Mmin)) #first index is for trajectory probability in the reduced space of x or y or z, second index is for upto different jump times in x,y,z
    indvproblist=np.zeros(3) #stores trajectory probability for the full trajectory in the reduced space of x/y/z
    jumplist=np.zeros((3,Mmin)) #stores the jump probabilities for each jump in x,y,z
    numlist=np.zeros(3) #stores the number of jumps in x,y,z
    waitprobtemp=np.zeros(3) #stores the waiting probability within each reduced space of x,y,z 

    for i in range(ind):
        t+=waits[i]
        if t>tfull: #only waiting probability is needed, as the next jump is after the end of the trajectory
            waitprob,_,_,waitprobtemp[0],waitprobtemp[1],waitprobtemp[2]=waitjumpprobxyz(xt[i],yt[i],zt[i],tfull-(t-waits[i]),xt[i+1],yt[i+1],zt[i+1])
            p+=waitprob
            indvproblist+=waitprobtemp
        else: #both waiting and jump probabilities are needed
            waitprob,jumpprob,jumpid,waitprobtemp[0],waitprobtemp[1],waitprobtemp[2]=waitjumpprobxyz(xt[i],yt[i],zt[i],waits[i],xt[i+1],yt[i+1],zt[i+1])
            p+=waitprob+jumpprob
            indvproblist+=waitprobtemp
            for j in range(3):
                problist[j,int(round(numlist[j]))]+=waitprob
                problist0[:,j,int(round(numlist[j]))]+=waitprobtemp
            if jumpid>-0.5: #this is a true jump
                indvproblist[jumpid]+=jumpprob
                jumplist[jumpid,int(round(numlist[jumpid]))]=jumpprob
                problist[jumpid,int(round(numlist[jumpid]))+1]=problist[jumpid,int(round(numlist[jumpid]))]+jumpprob
                problist0[:,jumpid,int(round(numlist[jumpid]))+1]=problist0[:,jumpid,int(round(numlist[jumpid]))]
                problist0[jumpid,jumpid,int(round(numlist[jumpid]))+1]+=jumpprob
                numlist[jumpid]+=1 #this gives the total number of jumps
            #The jumpid>-0.5 filter is needed because sometimes jumpid is -1, because the merged waiting times have jumps listed at phantom times when neither x,y,z change. This is because the merged waits list from a frozen x trajectory and a simulated yz trajectory, will also have the jumps from the frozen yz trajectories that accompanied the frozen x trajectory. The current way of handling it is the most numerically efficient, because otherwise the frozen x trajectory waiting list has to be separately filtered out.
                       
    return p,problist,indvproblist,problist0,jumplist,numlist 

#the underlying module is to calculate a steady-state histogram from which initial conditions can be sampled
datanx,datany,datanz,waittest,indtest=propagatexyz(int(round((kyd+kxy)/4.)),1,1,0.1,0,0,10000) #steady state trajectory
data=np.stack((datanx[:indtest],datany[:indtest],datanz[:indtest],waittest[:indtest]),axis=-1) #snapshots from trajectories after relaxation into steady-state
nx=11
ny=11
nz=11

#stationary distribution of x,y,z
Hxyz,elist=np.histogramdd(data[:,:3],bins=(nx,ny,nz),range=[(np.amin(data[:,0]),np.amax(data[:,0])),(np.amin(data[:,1]),np.amax(data[:,1])),(np.amin(data[:,2]),np.amax(data[:,2]))],weights=data[:,3],density=True)
#print(elist)
xe=elist[0]
ye=elist[1]
ze=elist[2]
xen=np.rint(0.5*(xe[1:]+xe[:-1])) #centers of the histogram bins
yen=np.rint(0.5*(ye[1:]+ye[:-1]))
zen=np.rint(0.5*(ze[1:]+ze[:-1]))

Hxyz/=np.sum(Hxyz)
Hxyzf=np.ndarray.flatten(Hxyz)

#P(y,z|x)
Hyz_x=np.zeros((nx,ny,nz))
Hyz_xf=np.zeros((nx,ny*nz))
for i in range(nx):
    Hyz_x[i,:,:]=Hxyz[i,:,:]
    if np.sum(Hyz_x[i,:,:])>tol:
        Hyz_x[i,:,:]/=np.sum(Hyz_x[i,:,:]) #Bayes' theorem
    else:
        Hyz_x[i,:,:]=0.
        Hyz_x[i,int(round(ny/2)),int(round(nz/2))]=1.
    Hyz_xf[i,:]=np.ndarray.flatten(Hyz_x[i,:,:])

#P(x,z|y)
Hxz_y=np.zeros((nx,ny,nz))
Hxz_yf=np.zeros((ny,nx*nz))
for j in range(ny):
    Hxz_y[:,j,:]=Hxyz[:,j,:]
    if np.sum(Hxz_y[:,j,:])>tol:
        Hxz_y[:,j,:]/=np.sum(Hxz_y[:,j,:]) #Bayes' theorem
    else:
        Hxz_y[:,j,:]=0.
        Hxz_y[int(round(nx/2)),j,int(round(nz/2))]=1.
    Hxz_yf[j,:]=np.ndarray.flatten(Hxz_y[:,j,:])

#P(x,y|z)
Hxy_z=np.zeros((nx,ny,nz))
Hxy_zf=np.zeros((nz,nx*ny))
for k in range(nz):
    Hxy_z[:,:,k]=Hxyz[:,:,k]
    if np.sum(Hxy_z[:,:,k])>tol:
        Hxy_z[:,:,k]/=np.sum(Hxy_z[:,:,k]) #Bayes' theorem
    else:
        Hxy_z[:,:,k]=0.
        Hxy_z[int(round(nx/2)),int(round(ny/2)),k]=1.
    Hxy_zf[k,:]=np.ndarray.flatten(Hxy_z[:,:,k])

#P(z|x,y)
Hz_xy=np.zeros((nx,ny,nz))
for i in range(nx):
    for j in range(ny):
        Hz_xy[i,j,:]=Hxyz[i,j,:]
        if np.sum(Hz_xy[i,j,:])>tol:
            Hz_xy[i,j,:]/=np.sum(Hz_xy[i,j,:]) #Bayes' theorem
        else:
            Hz_xy[i,j,:]=0.
            Hz_xy[i,j,int(round(nz/2))]=1.

#P(x|y,z)
Hx_yz=np.zeros((nx,ny,nz))
for j in range(ny):
    for k in range(nz):
        Hx_yz[:,j,k]=Hxyz[:,j,k]
        if np.sum(Hx_yz[:,j,k])>tol:
            Hx_yz[:,j,k]/=np.sum(Hx_yz[:,j,k]) #Bayes' theorem
        else:
            Hx_yz[:,j,k]=0.
            Hx_yz[int(round(nx/2)),j,k]=1.

#P(y|x,z)
Hy_xz=np.zeros((nx,ny,nz))
for i in range(nx):
    for k in range(nz):
        Hy_xz[i,:,k]=Hxyz[i,:,k]
        if np.sum(Hy_xz[i,:,k])>tol:
            Hy_xz[i,:,k]/=np.sum(Hy_xz[i,:,k]) #Bayes' theorem
        else:
            Hy_xz[i,:,k]=0.
            Hy_xz[i,int(round(ny/2)),k]=1.

#flatten all multidimensional histograms so that they can sampled from directly
xyzen=np.zeros((nx*ny*nz,3))
cnt=0
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            xyzen[cnt,0]=xen[i]
            xyzen[cnt,1]=yen[j]
            xyzen[cnt,2]=zen[k]
            cnt+=1
yzen=np.zeros((ny*nz,2))
cnt=0
for j in range(ny):
    for k in range(nz):
        yzen[cnt,0]=yen[j]
        yzen[cnt,1]=zen[k]
        cnt+=1
xzen=np.zeros((nx*nz,2))
cnt=0
for i in range(nx):
    for k in range(nz):
        xzen[cnt,0]=xen[i]
        xzen[cnt,1]=zen[k]
        cnt+=1
xyen=np.zeros((nx*ny,2))
cnt=0
for i in range(nx):
    for j in range(ny):
        xyen[cnt,0]=xen[i]
        xyen[cnt,1]=yen[j]
        cnt+=1

########### the following subroutines are for accessing various conditional distributions for computing marginal transition probabilities
@njit
def marginalx(xt,waitsfull,statesn,M): #given xt, compute marginal probability P(xt) through RR scheme (required for the second term in TE expression)
    steps=int(round(stepmags[-1])) #maximum number of steps
    yztrajs=np.zeros((K,2,2)) #holder for yz trajectory ensemble
    Uk=np.zeros(K) #holder for logarithm of joint probabilities
    Ukt=np.copy(Uk)
    Uktold=np.zeros(K)
    Umin=0.0 #required for the log-sum-exp trick #https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    Uminold=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K #metric for uniformity in the trajectory weights
    w=K
    Mcount=0
    fflag=0 #whether to use a given waiting time and the given identity of the next reaction to start the trajectory, set to 1 after non-cloning step, but 0 after cloning step 
    
    Kctyzn=0 #this counts the number of times resampling occurs, if this is too high then more trajectories are needed

    Mp=np.amax(np.array([Mmin,1])) #size of holder for reactions
    yn=np.zeros((K,Mp+1))
    zn=np.zeros((K,Mp+1))
    waits=np.zeros((K,Mp+1))
    inds=np.zeros(K) #number of states for yt and zt
    
    wait0ns=np.zeros(K) #for explanations of these four variables see the subroutine 'propagatexy' 
    Rins=np.zeros(K)
    indn=np.zeros(K) #index for xt
    waitcarry=np.zeros(K)+waitsfull[0]
    
    indn2=np.zeros(K) #for carrying the new indn and waitcarry after propagate
    waitcarry2=np.zeros(K) 
    
    xn2=np.zeros((K,Mp+1)) #for carrying merged state list from simulated and frozen trajectories
    yn2=np.zeros((K,Mp+1))
    zn2=np.zeros((K,Mp+1))
    mergedwaits=np.zeros((K,Mp+1))
    inds2=np.zeros(K) #number of states in merged trajectory

    problist=np.zeros((K,3,Mmin)) #for explanations of these five variables see the subroutine 'trajprob'
    problist0=np.zeros((K,3,3,Mmin))
    indvproblist=np.zeros((3,K))
    jumplist=np.zeros((K,3,Mmin))
    numlist=np.zeros((K,3))

    Tvals=np.zeros(M) #transfer entropy (J) estimator
    Tstepwise=0.
    Twvals=np.zeros(M) #waiting time contribution to the transfer entropy: (J+E) estimator minus (J) estimator
    Twstepwise=0.
    
    #set initial conditions from a conditioned histogram
    if xt[0]<=xe[0]:
        ix=0
    elif xt[0]>=xe[-1]:
        ix=nx-1
    else:
        ix=np.searchsorted(xe[1:],xt[0])
    yzN=np.zeros((K,2))
    yzN=yzen[np.searchsorted(np.cumsum(Hyz_xf[ix,:]),(np.random.random(K)+np.arange(K))/K,side="right")]
    yztrajs[:,0,0]=yzN[:,0]
    yztrajs[:,1,0]=yzN[:,1]
    yztrajs[:,:,1]=yztrajs[:,:,0]

    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            Kctyzn+=1
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            yztrajs[:,:,1]=(yztrajs[:,:,1])[np.searchsorted(np.cumsum(np.exp(-Ukt)/w), (np.random.random(K)+np.arange(K))/K, side="right")]
            #clocks and next reaction id are not resampled

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.
            
            #in the next step sample new clocks
            fflag=0
        elif i>0:
            fflag=1 #except the first step and the step right after resampling, reuse pre-sampled waiting times and identity of the next jump
            
        #propagate ensemble of x trajectories for one step and accumulate weight
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            yztrajs[k,:,0]=yztrajs[k,:,1]
            yn[k,:],zn[k,:],waits[k,:],inds[k],wait0ns[k],Rins[k],indn2[k],waitcarry2[k]=propagateyz(xt,yztrajs[k,0,0],yztrajs[k,1,0],wait0ns[k],int(round(Rins[k])),waitsfull,int(round(indn[k])),waitcarry[k],statesn,fflag,1)
            yztrajs[k,0,1]=yn[k,int(round(inds[k]))-1]
            yztrajs[k,1,1]=zn[k,int(round(inds[k]))-1]
            xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],inds2[k]=mergewaitsyz(xt,yn[k,:],zn[k,:],waitsfull,int(round(indn[k])),waitcarry[k],statesn,waits[k,:])
            Uj,problist[k,:,:],indvproblist[:,k],problist0[k,:,:,:],jumplist[k,:,:],numlist[k,:]=trajprob(xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],int(round(inds2[k])))
            Uj=-Uj
            U0=-indvproblist[1,k]-indvproblist[2,k] #if there are simultaneous jumps, these would not simply be the sum        
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #average the jump probabilities (not their logarithms) with the weights, then take log and sum over all jumps, to get (J) estimator increment
        for j in range(int(round(numlist[0,0]))): #loop through jumps of x
            problist[:,0,j]-=Uktold
            problist[:,0,j]-=problist0[:,1,0,j]+problist0[:,2,0,j] #trajectory probability in the reduced (y,z) space upto the different jump times in x
            probmax=np.amax(problist[:,0,j])
            problist[:,0,j]-=probmax
            jumpmax=np.amax(jumplist[:,0,j])
            jumplist[:,0,j]-=jumpmax #Again the log-sum-exp trick #https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
            Tstepwise+=np.log(np.dot(np.exp(jumplist[:,0,j]),np.exp(problist[:,0,j])))+jumpmax
            Tstepwise-=np.log(np.sum(np.exp(problist[:,0,j])))
            jumplist[:,0,j]+=jumpmax

        for k in range(K):
            indvproblist[0,k]-=np.sum(jumplist[k,0,:int(round(numlist[0,0]))]) #subtract the jump probabilities to get only the waiting time probabilities
        dTwstepwise=np.dot(indvproblist[0,:],np.exp(-Uktold)) #the estimation of average escape propensity with the quadrature approximation
        dTwstepwise/=np.sum(np.exp(-Uktold)) 
        Twstepwise+=dTwstepwise 

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)
        
        #update pointers
        indn=np.copy(indn2)
        waitcarry=np.copy(waitcarry2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin #Monte-Carlo average
            Tvals[Mcount]=Tstepwise
            Twvals[Mcount]=Twstepwise
            Mcount=Mcount+1

    return wval,Kctyzn,Tvals,Twvals

@njit
def marginaly(yt,waitsfull,statesn,M): #given yt, compute marginal probability P(yt) through RR scheme (required for the second term in TE expression)
#For explanations see the subroutine 'marginalx'
    steps=int(round(stepmags[-1])) 
    xztrajs=np.zeros((K,2,2))
    Uk=np.zeros(K)
    Ukt=np.copy(Uk)
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    wval=np.zeros(M)
    Keff=K
    w=K
    Mcount=0
    fflag=0 
    
    Kctxzn=0

    Mp=np.amax(np.array([Mmin,1])) 
    xn=np.zeros((K,Mp+1))
    zn=np.zeros((K,Mp+1))
    waits=np.zeros((K,Mp+1))
    inds=np.zeros(K) 
    wait0ns=np.zeros(K)
    Rins=np.zeros(K)
    indn=np.zeros(K)
    waitcarry=np.zeros(K)+waitsfull[0]
    indn2=np.zeros(K)
    waitcarry2=np.zeros(K) 
    xn2=np.zeros((K,Mp+1))
    yn2=np.zeros((K,Mp+1))
    zn2=np.zeros((K,Mp+1))
    mergedwaits=np.zeros((K,Mp+1))
    inds2=np.zeros(K)

    problist=np.zeros((K,3,Mmin))
    problist0=np.zeros((K,3,3,Mmin))
    indvproblist=np.zeros((3,K))
    jumplist=np.zeros((K,3,Mmin))
    numlist=np.zeros((K,3))

    Tvals=np.zeros(M) 
    Tstepwise=0.
    Twvals=np.zeros(M)
    Twstepwise=0.

    #set initial conditions from a conditioned histogram
    if yt[0]<=ye[0]:
        iy=0
    elif yt[0]>=ye[-1]:
        iy=ny-1
    else:
        iy=np.searchsorted(ye[1:],yt[0]) 
    xzN=np.zeros((K,2))
    xzN=xzen[np.searchsorted(np.cumsum(Hxz_yf[iy,:]),(np.random.random(K)+np.arange(K))/K,side="right")]
    xztrajs[:,0,0]=xzN[:,0]
    xztrajs[:,1,0]=xzN[:,1]
    xztrajs[:,:,1]=xztrajs[:,:,0]

    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            Kctxzn+=1
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            xztrajs[:,:,1]=(xztrajs[:,:,1])[np.searchsorted(np.cumsum(np.exp(-Ukt)/w), (np.random.random(K)+np.arange(K))/K, side="right")]

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.
            
            fflag=0
        elif i>0:
            fflag=1
            
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            xztrajs[k,:,0]=xztrajs[k,:,1]
            xn[k,:],zn[k,:],waits[k,:],inds[k],wait0ns[k],Rins[k],indn2[k],waitcarry2[k]=propagatexz(xztrajs[k,0,0],yt,xztrajs[k,1,0],wait0ns[k],int(round(Rins[k])),waitsfull,int(round(indn[k])),waitcarry[k],statesn,fflag,1)
            xztrajs[k,0,1]=xn[k,int(round(inds[k]))-1]
            xztrajs[k,1,1]=zn[k,int(round(inds[k]))-1]
            xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],inds2[k]=mergewaitsxz(xn[k,:],yt,zn[k,:],waitsfull,int(round(indn[k])),waitcarry[k],statesn,waits[k,:])
            Uj,problist[k,:,:],indvproblist[:,k],problist0[k,:,:,:],jumplist[k,:,:],numlist[k,:]=trajprob(xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],int(round(inds2[k])))
            Uj=-Uj
            U0=-indvproblist[0,k]-indvproblist[2,k] 
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #average the jump probabilities (not their logarithms) with the weights, then take log and sum over all jumps
        for j in range(int(round(numlist[0,1]))): #loop through jumps of y
            problist[:,1,j]-=Uktold
            problist[:,1,j]-=problist0[:,0,1,j]+problist0[:,2,1,j]
            probmax=np.amax(problist[:,1,j])
            problist[:,1,j]-=probmax
            jumpmax=np.amax(jumplist[:,1,j])
            jumplist[:,1,j]-=jumpmax
            Tstepwise+=np.log(np.dot(np.exp(jumplist[:,1,j]),np.exp(problist[:,1,j])))+jumpmax
            Tstepwise-=np.log(np.sum(np.exp(problist[:,1,j])))
            jumplist[:,1,j]+=jumpmax 

        for k in range(K):
            indvproblist[1,k]-=np.sum(jumplist[k,1,:int(round(numlist[0,1]))])
        dTwstepwise=np.dot(indvproblist[1,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold))
        Twstepwise+=dTwstepwise

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)
        
        #update pointers
        indn=np.copy(indn2)
        waitcarry=np.copy(waitcarry2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount]=Tstepwise
            Twvals[Mcount]=Twstepwise
            Mcount=Mcount+1

    return wval,Kctxzn,Tvals,Twvals

@njit
def marginalz(zt,waitsfull,statesn,M): #given zt, compute marginal probability P(zt) through RR scheme (required for the second term in TE expression)
#For explanations see the subroutine 'marginalx'
    steps=int(round(stepmags[-1])) 
    xytrajs=np.zeros((K,2,2)) 
    Uk=np.zeros(K)
    Ukt=np.copy(Uk)
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    wval=np.zeros(M) 
    Keff=K
    w=K
    Mcount=0
    fflag=0  
    
    Kctxyn=0

    Mp=np.amax(np.array([Mmin,1])) 
    xn=np.zeros((K,Mp+1))
    yn=np.zeros((K,Mp+1))
    waits=np.zeros((K,Mp+1))
    inds=np.zeros(K) 
    wait0ns=np.zeros(K)
    Rins=np.zeros(K)
    indn=np.zeros(K) 
    waitcarry=np.zeros(K)+waitsfull[0]
    indn2=np.zeros(K)
    waitcarry2=np.zeros(K) 
    xn2=np.zeros((K,Mp+1)) 
    yn2=np.zeros((K,Mp+1))
    zn2=np.zeros((K,Mp+1))
    mergedwaits=np.zeros((K,Mp+1))
    inds2=np.zeros(K)

    problist=np.zeros((K,3,Mmin))
    problist0=np.zeros((K,3,3,Mmin))
    indvproblist=np.zeros((3,K))
    jumplist=np.zeros((K,3,Mmin))
    numlist=np.zeros((K,3))

    Tvals=np.zeros(M) 
    Tstepwise=0.
    Twvals=np.zeros(M)
    Twstepwise=0.

    #set initial conditions from a conditioned histogram
    if zt[0]<=ze[0]:
        iz=0
    elif zt[0]>=ze[-1]:
        iz=nz-1
    else:
        iz=np.searchsorted(ze[1:],zt[0])
    xyN=np.zeros((K,2))
    xyN=xyen[np.searchsorted(np.cumsum(Hxy_zf[iz,:]),(np.random.random(K)+np.arange(K))/K,side="right")]
    xytrajs[:,0,0]=xyN[:,0]
    xytrajs[:,1,0]=xyN[:,1]
    xytrajs[:,:,1]=xytrajs[:,:,0]

    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            Kctxyn+=1
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            xytrajs[:,:,1]=(xytrajs[:,:,1])[np.searchsorted(np.cumsum(np.exp(-Ukt)/w), (np.random.random(K)+np.arange(K))/K, side="right")]

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.
            
            fflag=0
        elif i>0:
            fflag=1
            
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            xytrajs[k,:,0]=xytrajs[k,:,1]
            xn[k,:],yn[k,:],waits[k,:],inds[k],wait0ns[k],Rins[k],indn2[k],waitcarry2[k]=propagatexy(xytrajs[k,0,0],xytrajs[k,1,0],zt,wait0ns[k],int(round(Rins[k])),waitsfull,int(round(indn[k])),waitcarry[k],statesn,fflag,1)
            xytrajs[k,0,1]=xn[k,int(round(inds[k]))-1]
            xytrajs[k,1,1]=yn[k,int(round(inds[k]))-1]
            xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],inds2[k]=mergewaitsxy(xn[k,:],yn[k,:],zt,waitsfull,int(round(indn[k])),waitcarry[k],statesn,waits[k,:])
            Uj,problist[k,:,:],indvproblist[:,k],problist0[k,:,:,:],jumplist[k,:,:],numlist[k,:]=trajprob(xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],int(round(inds2[k])))
            Uj=-Uj
            U0=-indvproblist[0,k]-indvproblist[1,k] 
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #average the jump probabilities (not their logarithms) with the weights, then take log and sum over all jumps
        for j in range(int(round(numlist[0,2]))): #loop through jumps of z
            problist[:,2,j]-=Uktold
            problist[:,2,j]-=problist0[:,0,2,j]+problist0[:,1,2,j]
            probmax=np.amax(problist[:,2,j])
            problist[:,2,j]-=probmax
            jumpmax=np.amax(jumplist[:,2,j])
            jumplist[:,2,j]-=jumpmax
            Tstepwise+=np.log(np.dot(np.exp(jumplist[:,2,j]),np.exp(problist[:,2,j])))+jumpmax
            Tstepwise-=np.log(np.sum(np.exp(problist[:,2,j])))
            jumplist[:,2,j]+=jumpmax 

        for k in range(K): 
            indvproblist[2,k]-=np.sum(jumplist[k,2,:int(round(numlist[0,2]))]) 
        dTwstepwise=np.dot(indvproblist[2,:],np.exp(-Uktold)) 
        dTwstepwise/=np.sum(np.exp(-Uktold))
        Twstepwise+=dTwstepwise 

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)
        
        #update pointers
        indn=np.copy(indn2)
        waitcarry=np.copy(waitcarry2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount]=Tstepwise
            Twvals[Mcount]=Twstepwise
            Mcount=Mcount+1

    return wval,Kctxyn,Tvals,Twvals

@njit
def marginalxy(xt,yt,waitsfull,statesn,M): #given (xt,yt), compute marginal probability P(xt,yt) through RR scheme (required for the first term in TE expression)
#For explanations see the subroutine 'marginalx'
    steps=int(round(stepmags[-1])) 
    ztrajs=np.zeros((K,2)) 
    Uk=np.zeros(K)
    Ukt=np.copy(Uk)
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    wval=np.zeros(M) 
    Keff=K
    w=K
    Mcount=0
    fflag=0  
    
    Kctzn=0

    Mp=np.amax(np.array([Mmin,1])) 
    zn=np.zeros((K,Mp+1))
    waits=np.zeros((K,Mp+1))
    inds=np.zeros(K) 
    wait0ns=np.zeros(K)
    Rins=np.zeros(K)
    indn=np.zeros(K) 
    waitcarry=np.zeros(K)+waitsfull[0]
    indn2=np.zeros(K)
    waitcarry2=np.zeros(K) 
    xn2=np.zeros((K,Mp+1)) 
    yn2=np.zeros((K,Mp+1))
    zn2=np.zeros((K,Mp+1))
    mergedwaits=np.zeros((K,Mp+1))
    inds2=np.zeros(K)
    
    problist=np.zeros((K,3,Mmin))
    problist0=np.zeros((K,3,3,Mmin))
    indvproblist=np.zeros((3,K))
    jumplist=np.zeros((K,3,Mmin))
    numlist=np.zeros((K,3))
  
    Tvals=np.zeros((M,2)) 
    Tstepwise=np.zeros(2)
    Twvals=np.zeros((M,2))
    Twstepwise=np.zeros(2)

    #set initial conditions from a conditioned histogram
    if xt[0]<=xe[0]:
        ix=0
    elif xt[0]>=xe[-1]:
        ix=nx-1
    else:
        ix=np.searchsorted(xe[1:],xt[0]) 
    if yt[0]<=ye[0]:
        iy=0
    elif yt[0]>=ye[-1]:
        iy=ny-1
    else:
        iy=np.searchsorted(ye[1:],yt[0]) 
    zN=np.zeros(K)
    zN=zen[np.searchsorted(np.cumsum(Hz_xy[ix,iy,:]),(np.random.random(K)+np.arange(K))/K,side="right")]
    ztrajs[:,0]=zN
    ztrajs[:,1]=ztrajs[:,0]
    
    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            Kctzn+=1
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            zindices=np.searchsorted(np.cumsum(np.exp(-Ukt)/w), (np.random.random(K)+np.arange(K))/K, side="right")
            ztrajs[:,1]=(ztrajs[:,1])[zindices]            

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.
            
            fflag=0
        elif i>0:
            fflag=1
            
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            ztrajs[k,0]=ztrajs[k,1]
            zn[k,:],waits[k,:],inds[k],wait0ns[k],Rins[k],indn2[k],waitcarry2[k]=propagatez(xt,yt,ztrajs[k,0],wait0ns[k],int(round(Rins[k])),waitsfull,int(round(indn[k])),waitcarry[k],statesn,fflag,1)
            ztrajs[k,1]=zn[k,int(round(inds[k]))-1]
            xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],inds2[k]=mergewaitsz(xt,yt,zn[k,:],waitsfull,int(round(indn[k])),waitcarry[k],statesn,waits[k,:])
            Uj,problist[k,:,:],indvproblist[:,k],problist0[k,:,:,:],jumplist[k,:,:],numlist[k,:]=trajprob(xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],int(round(inds2[k])))
            Uj=-Uj
            U0=-indvproblist[2,k] 
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #contribution to transfer entropy
        #average the jump probabilities (not their logarithms) with the weights, then take log and sum over all jumps
        for j in range(int(round(numlist[0,0]))): #loop through jumps of x
            problist[:,0,j]-=Uktold
            problist[:,0,j]-=problist0[:,2,0,j]
            probmax=np.amax(problist[:,0,j])
            problist[:,0,j]-=probmax
            jumpmax=np.amax(jumplist[:,0,j])
            jumplist[:,0,j]-=jumpmax
            Tstepwise[0]+=np.log(np.dot(np.exp(jumplist[:,0,j]),np.exp(problist[:,0,j])))+jumpmax
            Tstepwise[0]-=np.log(np.sum(np.exp(problist[:,0,j])))
            jumplist[:,0,j]+=jumpmax
        for j in range(int(round(numlist[0,1]))): #loop through jumps of y
            problist[:,1,j]-=Uktold
            problist[:,1,j]-=problist0[:,2,1,j]
            probmax=np.amax(problist[:,1,j])
            problist[:,1,j]-=probmax
            jumpmax=np.amax(jumplist[:,1,j])
            jumplist[:,1,j]-=jumpmax
            Tstepwise[1]+=np.log(np.dot(np.exp(jumplist[:,1,j]),np.exp(problist[:,1,j])))+jumpmax
            Tstepwise[1]-=np.log(np.sum(np.exp(problist[:,1,j])))
            jumplist[:,1,j]+=jumpmax

        for k in range(K):
            indvproblist[0,k]-=np.sum(jumplist[k,0,:int(round(numlist[0,0]))])
            indvproblist[1,k]-=np.sum(jumplist[k,1,:int(round(numlist[0,1]))])
        dTwstepwise=np.dot(indvproblist[0,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold)) 
        Twstepwise[0]+=dTwstepwise 
        dTwstepwise=np.dot(indvproblist[1,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold)) 
        Twstepwise[1]+=dTwstepwise

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)
        
        #update pointers
        indn=np.copy(indn2)
        waitcarry=np.copy(waitcarry2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount,:]=Tstepwise
            Twvals[Mcount,:]=Twstepwise
            Mcount=Mcount+1

    return wval,Kctzn,Tvals,Twvals

@njit
def marginalyz(yt,zt,waitsfull,statesn,M): #given (yt,zt), compute marginal probability P(yt,zt) through RR scheme (required for the first term in TE expression)
#For explanations see the subroutine 'marginalx'
    steps=int(round(stepmags[-1])) 
    xtrajs=np.zeros((K,2)) 
    Uk=np.zeros(K)
    Ukt=np.copy(Uk)
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    wval=np.zeros(M) 
    Keff=K
    w=K
    Mcount=0
    fflag=0 
    
    Kctxn=0

    Mp=np.amax(np.array([Mmin,1]))
    xn=np.zeros((K,Mp+1))
    waits=np.zeros((K,Mp+1))
    inds=np.zeros(K) 
    wait0ns=np.zeros(K)
    Rins=np.zeros(K)
    indn=np.zeros(K) 
    waitcarry=np.zeros(K)+waitsfull[0]
    indn2=np.zeros(K)
    waitcarry2=np.zeros(K) 
    xn2=np.zeros((K,Mp+1)) 
    yn2=np.zeros((K,Mp+1))
    zn2=np.zeros((K,Mp+1))
    mergedwaits=np.zeros((K,Mp+1))
    inds2=np.zeros(K)
    
    problist=np.zeros((K,3,Mmin))
    problist0=np.zeros((K,3,3,Mmin))
    indvproblist=np.zeros((3,K))
    jumplist=np.zeros((K,3,Mmin))
    numlist=np.zeros((K,3))
  
    Tvals=np.zeros((M,2)) 
    Tstepwise=np.zeros(2)
    Twvals=np.zeros((M,2))
    Twstepwise=np.zeros(2)

    #set initial conditions from a conditioned histogram
    if yt[0]<=ye[0]:
        iy=0
    elif yt[0]>=ye[-1]:
        iy=ny-1
    else:
        iy=np.searchsorted(ye[1:],yt[0]) 
    if zt[0]<=ze[0]:
        iz=0
    elif zt[0]>=ze[-1]:
        iz=nz-1
    else:
        iz=np.searchsorted(ze[1:],zt[0]) 
    xN=np.zeros(K)
    xN=xen[np.searchsorted(np.cumsum(Hx_yz[:,iy,iz]),(np.random.random(K)+np.arange(K))/K,side="right")]
    xtrajs[:,0]=xN
    xtrajs[:,1]=xtrajs[:,0]
    
    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            Kctxn+=1
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            xindices=np.searchsorted(np.cumsum(np.exp(-Ukt)/w), (np.random.random(K)+np.arange(K))/K, side="right")
            xtrajs[:,1]=(xtrajs[:,1])[xindices]            

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.
            
            fflag=0
        elif i>0:
            fflag=1
            
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            xtrajs[k,0]=xtrajs[k,1]
            xn[k,:],waits[k,:],inds[k],wait0ns[k],Rins[k],indn2[k],waitcarry2[k]=propagatex(xtrajs[k,0],yt,zt,wait0ns[k],int(round(Rins[k])),waitsfull,int(round(indn[k])),waitcarry[k],statesn,fflag,1)
            xtrajs[k,1]=xn[k,int(round(inds[k]))-1]
            xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],inds2[k]=mergewaitsx(xn[k,:],yt,zt,waitsfull,int(round(indn[k])),waitcarry[k],statesn,waits[k,:])
            Uj,problist[k,:,:],indvproblist[:,k],problist0[k,:,:,:],jumplist[k,:,:],numlist[k,:]=trajprob(xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],int(round(inds2[k])))
            Uj=-Uj
            U0=-indvproblist[0,k] 
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #contribution to transfer entropy
        #average the jump probabilities (not their logarithms) with the weights, then take log and sum over all jumps
        for j in range(int(round(numlist[0,1]))): #loop through jumps of y
            problist[:,1,j]-=Uktold
            problist[:,1,j]-=problist0[:,0,1,j]
            probmax=np.amax(problist[:,1,j])
            problist[:,1,j]-=probmax
            jumpmax=np.amax(jumplist[:,1,j])
            jumplist[:,1,j]-=jumpmax
            Tstepwise[0]+=np.log(np.dot(np.exp(jumplist[:,1,j]),np.exp(problist[:,1,j])))+jumpmax
            Tstepwise[0]-=np.log(np.sum(np.exp(problist[:,1,j])))
            jumplist[:,1,j]+=jumpmax
        for j in range(int(round(numlist[0,2]))): #loop through jumps of z
            problist[:,2,j]-=Uktold
            problist[:,2,j]-=problist0[:,0,2,j]
            probmax=np.amax(problist[:,2,j])
            problist[:,2,j]-=probmax
            jumpmax=np.amax(jumplist[:,2,j])
            jumplist[:,2,j]-=jumpmax
            Tstepwise[1]+=np.log(np.dot(np.exp(jumplist[:,2,j]),np.exp(problist[:,2,j])))+jumpmax
            Tstepwise[1]-=np.log(np.sum(np.exp(problist[:,2,j])))
            jumplist[:,2,j]+=jumpmax 

        for k in range(K):
            indvproblist[1,k]-=np.sum(jumplist[k,1,:int(round(numlist[0,1]))])
            indvproblist[2,k]-=np.sum(jumplist[k,2,:int(round(numlist[0,2]))])
        dTwstepwise=np.dot(indvproblist[1,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold))
        Twstepwise[0]+=dTwstepwise
        dTwstepwise=np.dot(indvproblist[2,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold))
        Twstepwise[1]+=dTwstepwise

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)
        
        #update pointers
        indn=np.copy(indn2)
        waitcarry=np.copy(waitcarry2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount,:]=Tstepwise
            Twvals[Mcount,:]=Twstepwise
            Mcount=Mcount+1

    return wval,Kctxn,Tvals,Twvals

@njit
def marginalxz(xt,zt,waitsfull,statesn,M): #given (xt,zt), compute marginal probability P(xt,zt) through RR scheme (required for the first term in TE expression)
#For explanations see the subroutine 'marginalx'
    steps=int(round(stepmags[-1]))
    ytrajs=np.zeros((K,2))
    Uk=np.zeros(K)
    Ukt=np.copy(Uk)
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    wval=np.zeros(M)
    Keff=K
    w=K
    Mcount=0
    fflag=0 
    
    Kctyn=0

    Mp=np.amax(np.array([Mmin,1])) 
    yn=np.zeros((K,Mp+1))
    waits=np.zeros((K,Mp+1))
    inds=np.zeros(K) 
    wait0ns=np.zeros(K)
    Rins=np.zeros(K)
    indn=np.zeros(K) 
    waitcarry=np.zeros(K)+waitsfull[0]
    indn2=np.zeros(K)
    waitcarry2=np.zeros(K)
    xn2=np.zeros((K,Mp+1))
    yn2=np.zeros((K,Mp+1))
    zn2=np.zeros((K,Mp+1))
    mergedwaits=np.zeros((K,Mp+1))
    inds2=np.zeros(K)
    
    problist=np.zeros((K,3,Mmin))
    problist0=np.zeros((K,3,3,Mmin))
    indvproblist=np.zeros((3,K))
    jumplist=np.zeros((K,3,Mmin))
    numlist=np.zeros((K,3))
    
    Tvals=np.zeros((M,2)) 
    Tstepwise=np.zeros(2)
    Twvals=np.zeros((M,2))
    Twstepwise=np.zeros(2)

    #set initial conditions from a conditioned histogram
    if xt[0]<=xe[0]:
        ix=0
    elif xt[0]>=xe[-1]:
        ix=nx-1
    else:
        ix=np.searchsorted(xe[1:],xt[0]) 
    if zt[0]<=ze[0]:
        iz=0
    elif zt[0]>=ze[-1]:
        iz=nz-1
    else:
        iz=np.searchsorted(ze[1:],zt[0]) 
    yN=np.zeros(K)
    yN=yen[np.searchsorted(np.cumsum(Hy_xz[ix,:,iz]),(np.random.random(K)+np.arange(K))/K,side="right")]
    ytrajs[:,0]=yN
    ytrajs[:,1]=ytrajs[:,0]
    
    for i in range(steps):
        #check if resampling is needed
        if Keff<K/2:
            Kctyn+=1
            for j in range(Mcount,M):
                wval[j]+=np.log(w/K)-Umin

            #resample with weights exponential of -Ukt
            yindices=np.searchsorted(np.cumsum(np.exp(-Ukt)/w), (np.random.random(K)+np.arange(K))/K, side="right")
            ytrajs[:,1]=(ytrajs[:,1])[yindices]            

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.
            
            fflag=0
        elif i>0:
            fflag=1 
            
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            ytrajs[k,0]=ytrajs[k,1]
            yn[k,:],waits[k,:],inds[k],wait0ns[k],Rins[k],indn2[k],waitcarry2[k]=propagatey(xt,ytrajs[k,0],zt,wait0ns[k],int(round(Rins[k])),waitsfull,int(round(indn[k])),waitcarry[k],statesn,fflag,1)
            ytrajs[k,1]=yn[k,int(round(inds[k]))-1]
            xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],inds2[k]=mergewaitsy(xt,yn[k,:],zt,waitsfull,int(round(indn[k])),waitcarry[k],statesn,waits[k,:])
            Uj,problist[k,:,:],indvproblist[:,k],problist0[k,:,:,:],jumplist[k,:,:],numlist[k,:]=trajprob(xn2[k,:],yn2[k,:],zn2[k,:],mergedwaits[k,:],int(round(inds2[k])))
            Uj=-Uj
            U0=-indvproblist[1,k] 
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #contribution to transfer entropy
        #average the jump probabilities (not their logarithms) with the weights, then take log and sum over all jumps
        for j in range(int(round(numlist[0,0]))): #loop through jumps of x
            problist[:,0,j]-=Uktold
            problist[:,0,j]-=problist0[:,1,0,j]
            probmax=np.amax(problist[:,0,j])
            problist[:,0,j]-=probmax
            jumpmax=np.amax(jumplist[:,0,j])
            jumplist[:,0,j]-=jumpmax
            Tstepwise[0]+=np.log(np.dot(np.exp(jumplist[:,0,j]),np.exp(problist[:,0,j])))+jumpmax
            Tstepwise[0]-=np.log(np.sum(np.exp(problist[:,0,j])))
            jumplist[:,0,j]+=jumpmax
        for j in range(int(round(numlist[0,2]))): #loop through jumps of z
            problist[:,2,j]-=Uktold
            problist[:,2,j]-=problist0[:,1,2,j]
            probmax=np.amax(problist[:,2,j])
            problist[:,2,j]-=probmax
            jumpmax=np.amax(jumplist[:,2,j])
            jumplist[:,2,j]-=jumpmax
            Tstepwise[1]+=np.log(np.dot(np.exp(jumplist[:,2,j]),np.exp(problist[:,2,j])))+jumpmax
            Tstepwise[1]-=np.log(np.sum(np.exp(problist[:,2,j])))
            jumplist[:,2,j]+=jumpmax

        for k in range(K):
            indvproblist[0,k]-=np.sum(jumplist[k,0,:int(round(numlist[0,0]))])
            indvproblist[2,k]-=np.sum(jumplist[k,2,:int(round(numlist[0,2]))])
        dTwstepwise=np.dot(indvproblist[0,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold)) 
        Twstepwise[0]+=dTwstepwise 
        dTwstepwise=np.dot(indvproblist[2,:],np.exp(-Uktold))
        dTwstepwise/=np.sum(np.exp(-Uktold))
        Twstepwise[1]+=dTwstepwise
        
        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)
        
        #update pointers
        indn=np.copy(indn2)
        waitcarry=np.copy(waitcarry2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount,:]=Tstepwise
            Twvals[Mcount,:]=Twstepwise
            Mcount=Mcount+1

    return wval,Kctyn,Tvals,Twvals
    
#seed for random number is supplied through command line argument   
s=int(sys.argv[1])

########this is the master subroutine for sampling the logarithm in the TE expression for different pairs of trajectories. This subroutine calls marginalization subroutines.
@njit(parallel=True)
def rrpws(s):
    M=np.shape(stepmags)[0] #calculate transfer entropies for all timesteps 
    Mp=stepmags[-1]*Mmin #maximum holder size for number of jumps in a trajectory

    Iestxz=np.zeros((N,M)) #holder for mutual information
    Kctxyd=np.zeros(N) #holder for the number of resampling steps, purely for studying the efficiency of the algorithm 
    Kctyzd=np.zeros(N)
    Kctyd=np.zeros(N)
    Tvalsxz=np.zeros((N,M,2)) #holder for transfer entropies

    Iestxy=np.zeros((N,M))
    Kctxzd=np.zeros(N)
    Kctzd=np.zeros(N)
    Tvalsxy=np.zeros((N,M,2))

    Iestyz=np.zeros((N,M))
    Kctxd=np.zeros(N)
    Tvalsyz=np.zeros((N,M,2))
    
    Tvalsx=np.zeros((N,M))
    Tvalsy=np.zeros((N,M))
    Tvalsz=np.zeros((N,M))

    Twvalsx=np.zeros((N,M))
    Twvalsy=np.zeros((N,M))
    Twvalsz=np.zeros((N,M))
    Twvalsxy=np.zeros((N,M,2))
    Twvalsyz=np.zeros((N,M,2))
    Twvalsxz=np.zeros((N,M,2))

    xts=np.zeros((N,Mp+1)) #holder for outer trajectories
    yts=np.zeros((N,Mp+1))
    zts=np.zeros((N,Mp+1))
    waits=np.zeros((N,Mp+1))
    statesn=np.zeros(N)

    pxz=np.zeros((N,M)) #holder for trajectory probabilities
    pyz=np.zeros((N,M))
    pxy=np.zeros((N,M))

    px=np.zeros((N,M))
    py=np.zeros((N,M))
    pz=np.zeros((N,M))

    np.random.seed(s) #change the seed to change the statistical realizations, or eliminate this altogether for a random seed

    #initialize from steady-state distribution
    xyzN=np.zeros((N,3))
    xyzN=xyzen[np.searchsorted(np.cumsum(Hxyzf),(np.random.random(N)+np.arange(N))/N,side="right")]
    
    #propagate trajectories for the Monte-Carlo average outside the logarithm in TE expression
    for i in prange(N): #prange explicitly parallelizes with numba
        xts[i,:],yts[i,:],zts[i,:],waits[i,:],statesn[i]=propagatexyz(xyzN[i,0],xyzN[i,1],xyzN[i,2],0.1,0,0,M)

    for i in prange(N):
	#comment out irrelevant lines if transfer entropy in every possible combination is not needed
        pxz[i,:],Kctyd[i],Tvalsxz[i,:,:],Twvalsxz[i,:,:]=marginalxz(xts[i,:],zts[i,:],waits[i,:],int(round(statesn[i])),M) #probability, resampling number, first terms in TE expression from different estimatoes
        Iestxz[i,:]+=pxz[i,:] #numerator in MI expression
        pyz[i,:],Kctxd[i],Tvalsyz[i,:,:],Twvalsyz[i,:,:]=marginalyz(yts[i,:],zts[i,:],waits[i,:],int(round(statesn[i])),M) #probability, resampling number, first terms in TE expression from different estimatoes
        Iestyz[i,:]+=pyz[i,:] #numerator in MI expression
        pxy[i,:],Kctzd[i],Tvalsxy[i,:,:],Twvalsxy[i,:,:]=marginalxy(xts[i,:],yts[i,:],waits[i,:],int(round(statesn[i])),M) #probability, resampling number, first terms in TE expression from different estimatoes
        Iestxy[i,:]+=pxy[i,:] #numerator in MI expression

        px[i,:],Kctyzd[i],Tvalsx[i,:],Twvalsx[i,:]=marginalx(xts[i,:],waits[i,:],int(round(statesn[i])),M)
        py[i,:],Kctxzd[i],Tvalsy[i,:],Twvalsy[i,:]=marginaly(yts[i,:],waits[i,:],int(round(statesn[i])),M)
        pz[i,:],Kctxyd[i],Tvalsz[i,:],Twvalsz[i,:]=marginalz(zts[i,:],waits[i,:],int(round(statesn[i])),M)

        Iestxz[i,:]-=px[i,:]+pz[i,:] #MI denominator
        Tvalsxz[i,:,0]-=Tvalsx[i,:] #Transfer entropy (J) estimator first minus second term
        Tvalsxz[i,:,1]-=Tvalsz[i,:] #Transfer entropy (J) estimator first minus second term
        Twvalsxz[i,:,0]-=Twvalsx[i,:] #Transfer entropy (J+E) estimator but only the (E) part, first minus second term
        Twvalsxz[i,:,1]-=Twvalsz[i,:] #Transfer entropy (J+E) estimator but only the (E) part, first minus second term
        Iestxy[i,:]-=px[i,:]+py[i,:] #MI denominator
        Tvalsxy[i,:,0]-=Tvalsx[i,:] #Transfer entropy (J) estimator first minus second term
        Tvalsxy[i,:,1]-=Tvalsy[i,:] #Transfer entropy (J) estimator first minus second term
        Twvalsxy[i,:,0]-=Twvalsx[i,:] #Transfer entropy (J+E) estimator but only the (E) part, first minus second term
        Twvalsxy[i,:,1]-=Twvalsy[i,:] #Transfer entropy (J+E) estimator but only the (E) part, first minus second term
        Iestyz[i,:]-=py[i,:]+pz[i,:] #MI denominator
        Tvalsyz[i,:,0]-=Tvalsy[i,:] #Transfer entropy (J) estimator first minus second term
        Tvalsyz[i,:,1]-=Tvalsz[i,:] #Transfer entropy (J) estimator first minus second term
        Twvalsyz[i,:,0]-=Twvalsy[i,:] #Transfer entropy (J+E) estimator but only the (E) part, first minus second term
        Twvalsyz[i,:,1]-=Twvalsz[i,:] #Transfer entropy (J+E) estimator but only the (E) part, first minus second term
        
        Twvalsxz[i,:,:]+=Tvalsxz[i,:,:] #Transfer entropy (J+E) estimator where (J) has been added to (E)
        Twvalsyz[i,:,:]+=Tvalsyz[i,:,:] #Transfer entropy (J+E) estimator where (J) has been added to (E)
        Twvalsxy[i,:,:]+=Tvalsxy[i,:,:] #Transfer entropy (J+E) estimator where (J) has been added to (E)

    return Iestxy,Iestyz,Iestxz,Kctxd,Kctyd,Kctzd,Kctxyd,Kctyzd,Kctxzd,Tvalsxy,Tvalsyz,Tvalsxz,Twvalsxy,Twvalsyz,Twvalsxz

#s=7
Iestxy,Iestyz,Iestxz,Kctxd,Kctyd,Kctzd,Kctxyd,Kctyzd,Kctxzd,Tvalsxy,Tvalsyz,Tvalsxz,Twvalsxy,Twvalsyz,Twvalsxz=rrpws(s)
Kct=np.stack((Kctxd,Kctyd,Kctzd,Kctxyd,Kctyzd,Kctxzd),axis=-1)

#Monte-Carlo average outside the logarithm in the TE expression
iest2xy=np.mean(Iestxy,axis=0)
iest2yz=np.mean(Iestyz,axis=0)
iest2xz=np.mean(Iestxz,axis=0)
tvals2xy=np.mean(Tvalsxy,axis=0)
tvals2yz=np.mean(Tvalsyz,axis=0)
tvals2xz=np.mean(Tvalsxz,axis=0)
twvals2xy=np.mean(Twvalsxy,axis=0)
twvals2yz=np.mean(Twvalsyz,axis=0)
twvals2xz=np.mean(Twvalsxz,axis=0)

#save cumulative information and TE for increasing number of timesteps, and resampling numbers for each trajectory
np.savetxt('Iestxy_'+str(s)+'.txt',iest2xy)
np.savetxt('Iestyz_'+str(s)+'.txt',iest2yz)
np.savetxt('Iestxz_'+str(s)+'.txt',iest2xz)
np.savetxt('Kct_'+str(s)+'.txt',Kct,fmt='%i')
np.savetxt('Tvalszx_'+str(s)+'.txt',tvals2xz[:,0])
np.savetxt('Tvalsxz_'+str(s)+'.txt',tvals2xz[:,1])
np.savetxt('Tvalsyx_'+str(s)+'.txt',tvals2xy[:,0])
np.savetxt('Tvalsxy_'+str(s)+'.txt',tvals2xy[:,1])
np.savetxt('Tvalszy_'+str(s)+'.txt',tvals2yz[:,0])
np.savetxt('Tvalsyz_'+str(s)+'.txt',tvals2yz[:,1])
np.savetxt('Twvalszx_'+str(s)+'.txt',twvals2xz[:,0])
np.savetxt('Twvalsxz_'+str(s)+'.txt',twvals2xz[:,1])
np.savetxt('Twvalsyx_'+str(s)+'.txt',twvals2xy[:,0])
np.savetxt('Twvalsxy_'+str(s)+'.txt',twvals2xy[:,1])
np.savetxt('Twvalszy_'+str(s)+'.txt',twvals2yz[:,0])
np.savetxt('Twvalsyz_'+str(s)+'.txt',twvals2yz[:,1])

