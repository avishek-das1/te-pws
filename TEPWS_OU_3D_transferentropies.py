#Algorithm for Transfer Entropy- Path Weight Sampling (TE-PWS) in a 3-dimensional OU process where den
#Output is the Monte-Carlo estimate of MI and all transfer entropies
#Written by Avishek Das, Aug 30, 2024
#For different expressions for drifts, change only the forcex, forcey, forcez, force0x, force0y, force0z subroutines

import numpy as np
import sys
import math
from numba import jit,njit,prange

s=int(sys.argv[1])#seed for random number, provided as command-line input #seed only works for the master thread, not any of the other threads when numba is being used, which is by default true

#Global variables
#3D Ornstein-Uhlenbeck process parameters: spring and diffusion constants matrix
kmat=np.array([[0.2093,-0.069,0.299],[-0.08449,0.2817,0.2093],[0.,0.,0.33]])
Dmat=np.array([[0.00996831,0.00697782,0.],[0.00697782,0.0293045,0.008448],[0.,0.008448,0.008448]])

#inverses and determinants will be needed for calculating the Onsager-Machlup action
Dinv=np.linalg.inv(Dmat)
Ddet=np.linalg.det(Dmat)

#Cholesky decomposition of diffusion constant matrix is needed in order to propagate the dynamics
#sigmamat will multiply standard normal noises
sigmamat=np.linalg.cholesky(2.*Dmat)

#diffusion matrix, inverses, determinants, and Cholesky decompositions in reduced spaces
Dxymat=np.array([[Dmat[0,0],Dmat[0,1]],[Dmat[1,0],Dmat[1,1]]])
Dyzmat=np.array([[Dmat[1,1],Dmat[1,2]],[Dmat[2,1],Dmat[2,2]]])
Dxzmat=np.array([[Dmat[0,0],Dmat[0,2]],[Dmat[2,0],Dmat[2,2]]])
Dxyinv=np.linalg.inv(Dxymat)
Dyzinv=np.linalg.inv(Dyzmat)
Dxzinv=np.linalg.inv(Dxzmat)
Dxydet=np.linalg.det(Dxymat)
Dyzdet=np.linalg.det(Dyzmat)
Dxzdet=np.linalg.det(Dxzmat)

sigmaxy=np.linalg.cholesky(2.*Dxymat)
sigmayz=np.linalg.cholesky(2.*Dyzmat)
sigmaxz=np.linalg.cholesky(2.*Dxzmat)

#conditional noise distribution related matrices
#Dx_yz refers to conditioned noise matrix for x given yz
#Sx_yz refers to the scaling matrix for the conditioned mean observable
#First one is for x given yz
Sx_yz=np.transpose(np.array([Dmat[0,1],Dmat[0,2]]))@Dyzinv
Dx_yz=Dmat[0,0]-np.transpose(np.array([Dmat[0,1],Dmat[0,2]]))@Dyzinv@np.array([Dmat[1,0],Dmat[2,0]])

Sy_xz=np.transpose(np.array([Dmat[1,0],Dmat[1,2]]))@Dxzinv
Dy_xz=Dmat[1,1]-np.transpose(np.array([Dmat[1,0],Dmat[1,2]]))@Dxzinv@np.array([Dmat[0,1],Dmat[2,1]])

Sz_xy=np.transpose(np.array([Dmat[2,0],Dmat[2,1]]))@Dxyinv
Dz_xy=Dmat[2,2]-np.transpose(np.array([Dmat[2,0],Dmat[2,1]]))@Dxyinv@np.array([Dmat[0,2],Dmat[1,2]])

Sxy_z=np.array([Dmat[0,2],Dmat[1,2]])/Dmat[2,2]
Dxy_z=Dxymat-np.outer(np.array([Dmat[0,2],Dmat[1,2]]),np.array([Dmat[2,0],Dmat[2,1]]))/Dmat[2,2]

Syz_x=np.array([Dmat[1,0],Dmat[2,0]])/Dmat[0,0]
Dyz_x=Dyzmat-np.outer(np.array([Dmat[1,0],Dmat[2,0]]),np.array([Dmat[0,1],Dmat[0,2]]))/Dmat[0,0]

Sxz_y=np.array([Dmat[0,1],Dmat[2,1]])/Dmat[1,1]
Dxz_y=Dxzmat-np.outer(np.array([Dmat[0,1],Dmat[2,1]]),np.array([Dmat[1,0],Dmat[1,2]]))/Dmat[1,1]

#inverses, determinants and Cholesky decompositions of conditioned noise matrices
Dx_yzinv=1./(Dx_yz)
Dy_xzinv=1./(Dy_xz)
Dz_xyinv=1./(Dz_xy)
Dxy_zinv=np.linalg.inv(Dxy_z)
Dxz_yinv=np.linalg.inv(Dxz_y)
Dyz_xinv=np.linalg.inv(Dyz_x)

Dx_yzdet=Dx_yz
Dy_xzdet=Dy_xz
Dz_xydet=Dz_xy
Dxy_zdet=np.linalg.det(Dxy_z)
Dxz_ydet=np.linalg.det(Dxz_y)
Dyz_xdet=np.linalg.det(Dyz_x)

sigmax_yz=(2.*Dx_yz)**0.5
sigmay_xz=(2.*Dy_xz)**0.5
sigmaz_xy=(2.*Dz_xy)**0.5
sigmaxy_z=np.linalg.cholesky(2.*Dxy_z)
sigmaxz_y=np.linalg.cholesky(2.*Dxz_y)
sigmayz_x=np.linalg.cholesky(2.*Dyz_x)

#timestep for Langevin dynamics and RR scheme
dt=0.0004

#number of timesteps
stepmags=np.arange(1,80001,1)

N=96 #=M_{1} in manuscript=number of trajectories for the outer Monte-Carlo average
K=200 #=M_{2} in manuscript=number of trajectories for marginalization

####force functions
@njit
def forcex(x,y,z): #force on x
    return -kmat[0,0]*x-kmat[0,1]*y-kmat[0,2]*z

@njit
def forcey(x,y,z): #force on y
    return -kmat[1,1]*y-kmat[1,0]*x-kmat[1,2]*z

@njit
def forcez(x,y,z): #force on z
    return -kmat[2,2]*z-kmat[2,1]*y-kmat[2,0]*x

####reference dynamics, aka choice of P0, kept identical to the above force functions
@njit
def force0x(x,y,z): #force on x
    return -kmat[0,0]*x-kmat[0,1]*y-kmat[0,2]*z

@njit
def force0y(x,y,z): #force on y
    return -kmat[1,1]*y-kmat[1,0]*x-kmat[1,2]*z

@njit
def force0z(x,y,z): #force on z
    return -kmat[2,2]*z-kmat[2,1]*y-kmat[2,0]*x

@njit
def propagatexyz(xi,yi,zi,steps): #langevin equation to propagate x,y,z from initial conditions xi,yi,zi for 'steps' number of steps
    xt=np.zeros(steps+1)
    yt=np.zeros(steps+1)
    zt=np.zeros(steps+1)
    xt[0]=xi
    yt[0]=yi
    zt[0]=zi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

	#compute noises, compute forces, combine
        psir=np.random.randn(3)
        fr=np.array([forcex(x,y,z),forcey(x,y,z),forcez(x,y,z)])
        dr=fr*dt+(dt)**0.5*sigmamat@psir

        xt[i+1]=x+dr[0]
        yt[i+1]=y+dr[1]
        zt[i+1]=z+dr[2]

    return xt,yt,zt

#the underlying module is to calculate a steady-state histogram from which initial conditions can be sampled
tol=1e-5 #tolerance for setting histogram entries to 0
datanx,datany,datanz=propagatexyz(0,0,0,int(1e7)) #steady-state trajectory
data=np.stack((datanx[1000::100],datany[1000::100],datanz[1000::100]),axis=-1) #snapshots from trajectories after relaxation into steady-state
nx=11
ny=11
nz=11

#stationary distribution of x,y,z
Hxyz,elist=np.histogramdd(data,bins=(nx,ny,nz),range=[(np.amin(data[:,0]),np.amax(data[:,0])),(np.amin(data[:,1]),np.amax(data[:,1])),(np.amin(data[:,2]),np.amax(data[:,2]))],density=True)
#print(elist)
xe=elist[0]
ye=elist[1]
ze=elist[2]
xen=0.5*(xe[1:]+xe[:-1])
yen=0.5*(ye[1:]+ye[:-1])
zen=0.5*(ze[1:]+ze[:-1]) #centers of the histogram bins

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
        Hyz_x[i,int(ny/2),int(nz/2)]=1.
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
        Hxz_y[int(nx/2),j,int(nz/2)]=1.
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
        Hxy_z[int(nx/2),int(ny/2),k]=1.
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
            Hz_xy[i,j,int(nz/2)]=1.

#P(x|y,z)
Hx_yz=np.zeros((nx,ny,nz))
for j in range(ny):
    for k in range(nz):
        Hx_yz[:,j,k]=Hxyz[:,j,k]
        if np.sum(Hx_yz[:,j,k])>tol:
            Hx_yz[:,j,k]/=np.sum(Hx_yz[:,j,k]) #Bayes' theorem
        else:
            Hx_yz[:,j,k]=0.
            Hx_yz[int(nx/2),j,k]=1.

#P(y|x,z)
Hy_xz=np.zeros((nx,ny,nz))
for i in range(nx):
    for k in range(nz):
        Hy_xz[i,:,k]=Hxyz[i,:,k]
        if np.sum(Hy_xz[i,:,k])>tol:
            Hy_xz[i,:,k]/=np.sum(Hy_xz[i,:,k]) #Bayes' theorem
        else:
            Hy_xz[i,:,k]=0.
            Hy_xz[i,int(ny/2),k]=1.

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

##########The below modules are for simulating the dynamics in various reduced spaces in the frozen field of a given observable
@njit
def propagatexy(xi,yi,zt,steps): #simulate x and y from initial conditions xi and yi in the frozen field of the given zt trajectory
    xt=np.zeros(steps+1)
    yt=np.zeros(steps+1)
    xt[0]=xi
    yt[0]=yi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        psir=np.random.randn(2)
        fr=np.array([force0x(x,y,z),force0y(x,y,z)])
        dr=fr*dt+(dt)**0.5*(sigmaxy_z@psir+Sxy_z*((zt[i+1]-zt[i]-forcez(x,y,z)*dt)/dt**0.5)) 
        #Above line is an application of Schur Lemma for conditional noise distribution. The conditional noise here for updating the future (x,y) has a variance dependent on the approximate noise in z, which is computed from the z trajectory and the current (x,y) values #all underlying subroutines are exactly analogous

        xt[i+1]=x+dr[0]
        yt[i+1]=y+dr[1]

    return xt,yt

@njit
def propagatexz(xi,yt,zi,steps): #simulate x and z from initial conditions xi and zi in the frozen field of the given yt trajectory
    xt=np.zeros(steps+1)
    zt=np.zeros(steps+1)
    xt[0]=xi
    zt[0]=zi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        psir=np.random.randn(2)
        fr=np.array([force0x(x,y,z),force0z(x,y,z)])
        dr=fr*dt+(dt)**0.5*(sigmaxz_y@psir+Sxz_y*((yt[i+1]-yt[i]-forcey(x,y,z)*dt)/dt**0.5)) #for explanation see 'propagatexy' subroutine 

        xt[i+1]=x+dr[0]
        zt[i+1]=z+dr[1]

    return xt,zt

@njit
def propagateyz(xt,yi,zi,steps): #simulate y and z from initial conditions yi and zi in the frozen field of the given xt trajectory
    yt=np.zeros(steps+1)
    zt=np.zeros(steps+1)
    yt[0]=yi
    zt[0]=zi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        psir=np.random.randn(2)
        fr=np.array([force0y(x,y,z),force0z(x,y,z)])
        dr=fr*dt+(dt)**0.5*(sigmayz_x@psir+Syz_x*((xt[i+1]-xt[i]-forcex(x,y,z)*dt)/dt**0.5)) #for explanation see 'propagatexy' subroutine 

        yt[i+1]=y+dr[0]
        zt[i+1]=z+dr[1]

    return yt,zt

@njit
def propagatez(xt,yt,zi,steps): #simulate z from initial conditions zi in the frozen field of the given (xt,yt) trajectories
    zt=np.zeros(steps+1)
    zt[0]=zi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fz=force0z(x,y,z)
        psi3=np.random.normal()
        dz=fz*dt+(dt)**0.5*(sigmaz_xy*psi3+Sz_xy@np.array([(xt[i+1]-xt[i]-forcex(x,y,z)*dt)/dt**0.5,(yt[i+1]-yt[i]-forcey(x,y,z)*dt)/dt**0.5])) #for explanation see 'propagatexy' subroutine 

        zt[i+1]=z+dz

    return zt

@njit
def propagatex(xi,yt,zt,steps): #simulate x from initial conditions xi in the frozen field of the given (yt,zt) trajectories
    xt=np.zeros(steps+1)
    xt[0]=xi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fx=force0x(x,y,z)
        psi=np.random.normal()
        dx=fx*dt+(dt)**0.5*(sigmax_yz*psi+Sx_yz@np.array([(yt[i+1]-yt[i]-forcey(x,y,z)*dt)/dt**0.5,(zt[i+1]-zt[i]-forcez(x,y,z)*dt)/dt**0.5])) #for explanation see 'propagatexy' subroutine 

        xt[i+1]=x+dx

    return xt

@njit
def propagatey(xt,yi,zt,steps): #simulate y from initial conditions yi in the frozen field of the given (xt,zt) trajectories
    yt=np.zeros(steps+1)
    yt[0]=yi

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fy=force0y(x,y,z)
        psi2=np.random.normal()
        dy=fy*dt+(dt)**0.5*(sigmay_xz*psi2+Sy_xz@np.array([(xt[i+1]-xt[i]-forcex(x,y,z)*dt)/dt**0.5,(zt[i+1]-zt[i]-forcez(x,y,z)*dt)/dt**0.5])) #for explanation see 'propagatexy' subroutine 

        yt[i+1]=y+dy

    return yt

##########The following subroutines are for computing trajectory probabilities in different dimensional spaces. The first is in the full space.
@njit
def trajprob(xt,yt,zt,M,stepmags): #given three trajectories it gives transition probabilities for each coordinate and the joint transition probability
    p=np.zeros((M,4)) #3 individual transition probabilties and one joint transition probability all in log space, cumulatively added up each step upto M steps
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fr=np.array([forcex(x,y,z),forcey(x,y,z),forcez(x,y,z)])
        deta=np.array([xt[i+1]-xt[i]-fr[0]*dt,yt[i+1]-yt[i]-fr[1]*dt,zt[i+1]-zt[i]-fr[2]*dt])
        pstep=-0.5*(np.log(Ddet*(4.*np.pi*dt)**3)+np.transpose(deta)@Dinv@deta/2./dt) #log of joint transition probability 

        pstep1=(-(xt[i+1]-xt[i]-fr[0]*dt)**2/4/Dmat[0,0]/dt)-np.log((4*np.pi*Dmat[0,0]*dt)**0.5) #log of individual transition probabilities
        pstep2=(-(yt[i+1]-yt[i]-fr[1]*dt)**2/4/Dmat[1,1]/dt)-np.log((4*np.pi*Dmat[1,1]*dt)**0.5)
        pstep3=(-(zt[i+1]-zt[i]-fr[2]*dt)**2/4/Dmat[2,2]/dt)-np.log((4*np.pi*Dmat[2,2]*dt)**0.5)

        for j in range(M):
            if i<stepmags[j]:
                p[j,0]+=pstep1
                p[j,1]+=pstep2
                p[j,2]+=pstep3
                p[j,3]+=pstep
    return p

#for explanations see 'trajprob' subroutine
@njit
def trajprob0z(xt,yt,zt,M,stepmags): #given zt trajectory it gives reference probability P_0(zt)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]
        fz=force0z(x,y,z)

        mu=Sz_xy@np.array([(xt[i+1]-xt[i]-forcex(x,y,z)*dt)/dt**0.5,(yt[i+1]-yt[i]-forcey(x,y,z)*dt)/dt**0.5])
        deta=(zt[i+1]-zt[i]-fz*dt-mu*dt**0.5)
        pstep=-0.5*(np.log(Dz_xydet*(4.*np.pi*dt)**1)+deta**2*Dz_xyinv/2./dt)

        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

#for explanations see 'trajprob' subroutine
@njit
def trajprob0x(xt,yt,zt,M,stepmags): #given xt trajectory it gives reference probability P_0(xt)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]
        fx=force0x(x,y,z)
        
        mu=Sx_yz@np.array([(yt[i+1]-yt[i]-forcey(x,y,z)*dt)/dt**0.5,(zt[i+1]-zt[i]-forcez(x,y,z)*dt)/dt**0.5])
        deta=(xt[i+1]-xt[i]-fx*dt-mu*dt**0.5)
        pstep=-0.5*(np.log(Dx_yzdet*(4.*np.pi*dt)**1)+deta**2*Dx_yzinv/2./dt)

        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

#for explanations see 'trajprob' subroutine
@njit
def trajprob0y(xt,yt,zt,M,stepmags): #given yt trajectory it gives reference probability P_0(yt)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]
        fy=force0y(x,y,z)

        mu=Sy_xz@np.array([(xt[i+1]-xt[i]-forcex(x,y,z)*dt)/dt**0.5,(zt[i+1]-zt[i]-forcez(x,y,z)*dt)/dt**0.5])
        deta=(yt[i+1]-yt[i]-fy*dt-mu*dt**0.5)
        pstep=-0.5*(np.log(Dy_xzdet*(4.*np.pi*dt)**1)+deta**2*Dy_xzinv/2./dt)

        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

#for explanations see 'trajprob' subroutine
@njit
def trajprob0xy(xt,yt,zt,M,stepmags): #given xt and yt trajectories it gives reference probability P_0(xt,yt)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fr=np.array([force0x(x,y,z),force0y(x,y,z)])
        mu=Sxy_z*(zt[i+1]-zt[i]-forcez(x,y,z)*dt)/dt**0.5
        deta=np.array([(xt[i+1]-xt[i]-fr[0]*dt-mu[0]*dt**0.5),(yt[i+1]-yt[i]-fr[1]*dt-mu[1]*dt**0.5)])
        pstep=-0.5*(np.log(Dxy_zdet*(4.*np.pi*dt)**2)+np.transpose(deta)@Dxy_zinv@deta/2./dt)

        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

#for explanations see 'trajprob' subroutine
@njit
def trajprob0yz(xt,yt,zt,M,stepmags): #given yt and zt trajectories it gives reference probability P_0(yt,zt)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fr=np.array([force0y(x,y,z),force0z(x,y,z)])
        mu=Syz_x*(xt[i+1]-xt[i]-forcex(x,y,z)*dt)/dt**0.5
        deta=np.array([(yt[i+1]-yt[i]-fr[0]*dt-mu[0]*dt**0.5),(zt[i+1]-zt[i]-fr[1]*dt-mu[1]*dt**0.5)])
        pstep=-0.5*(np.log(Dyz_xdet*(4.*np.pi*dt)**2)+np.transpose(deta)@Dyz_xinv@deta/2./dt)

        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

#for explanations see 'trajprob' subroutine
@njit
def trajprob0xz(xt,yt,zt,M,stepmags): #given xt and zt it gives reference probability P_0(xt,zt)
    p=np.zeros(M)
    steps=int(stepmags[-1]) #maximum number of steps

    for i in range(steps):
        x=xt[i]
        y=yt[i]
        z=zt[i]

        fr=np.array([force0x(x,y,z),force0z(x,y,z)])
        mu=Sxz_y*(yt[i+1]-yt[i]-forcey(x,y,z)*dt)/dt**0.5
        deta=np.array([(xt[i+1]-xt[i]-fr[0]*dt-mu[0]*dt**0.5),(zt[i+1]-zt[i]-fr[1]*dt-mu[1]*dt**0.5)])
        pstep=-0.5*(np.log(Dxz_ydet*(4.*np.pi*dt)**2)+np.transpose(deta)@Dxz_yinv@deta/2./dt)

        for j in range(M):
            if i<stepmags[j]:
                p[j]+=pstep
    return p

########### the following subroutines are for accessing various conditional distributions for computing marginal transition probabilities
@njit
def marginalx(xt,M,stepmags,K): #given xt, compute marginal probability P(xt) through RR scheme (required for the denominator in TE expression)
    steps=int(stepmags[-1]) #maximum number of steps
    yztrajs=np.zeros((K,2,2)) #holder for yz trajectory ensemble
    Uk=np.zeros(K) #holder for logarithm of joint probabilities
    Ukt=Uk
    Umin=0.0 #required for the log-sum-exp trick #https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K #metric for uniformity in the trajectory weights
    w=K
    Mcount=0
    
    Kctyzn=0 #this counts the number of times resampling occurs, if this is too high then more trajectories are needed

    yn=np.zeros((K,2))
    zn=np.zeros((K,2))

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

            #reset Uk
            Uk[:]=0.
            
        #propagate ensemble of y trajectories for one step and accumulate weight
        for k in range(K):
            yztrajs[k,:,0]=yztrajs[k,:,1]
            yn[k,:],zn[k,:]=propagateyz(xt[i:i+2],yztrajs[k,0,0],yztrajs[k,1,0],1)
            yztrajs[k,0,1]=yn[k,-1]
            yztrajs[k,1,1]=zn[k,-1]
            Uj=-trajprob(xt[i:i+2],yztrajs[k,0,:],yztrajs[k,1,:],1,np.array([1]))[-1,-1]
            U0=-trajprob0yz(xt[i:i+2],yztrajs[k,0,:],yztrajs[k,1,:],1,np.array([1]))[-1]
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity #aka the log-sum-exp trick

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin #Monte Carlo average
            Mcount=Mcount+1

    return wval,Kctyzn #wval is the logarithm of the cumulative marginal probability upto M consecutive timesteps, and Kctyzn is the number of times resampling occurs

#for detailed comments see subroutine 'marginalx'
@njit
def marginaly(yt,M,stepmags,K): #given yt, compute marginal through RR method
    steps=int(stepmags[-1]) #maximum number of steps
    xztrajs=np.zeros((K,2,2))
    Uk=np.zeros(K)
    Ukt=Uk
    Umin=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0
    
    Kctxzn=0

    xn=np.zeros((K,2))
    zn=np.zeros((K,2))

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
            
        #propagate ensemble of y trajectories for one step and accumulate weight
        for k in range(K):
            xztrajs[k,:,0]=xztrajs[k,:,1]
            xn[k,:],zn[k,:]=propagatexz(xztrajs[k,0,0],yt[i:i+2],xztrajs[k,1,0],1)
            xztrajs[k,0,1]=xn[k,-1]
            xztrajs[k,1,1]=zn[k,-1]
            Uj=-trajprob(xztrajs[k,0,:],yt[i:i+2],xztrajs[k,1,:],1,np.array([1]))[-1,-1]
            U0=-trajprob0xz(xztrajs[k,0,:],yt[i:i+2],xztrajs[k,1,:],1,np.array([1]))[-1]
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Mcount=Mcount+1

    return wval,Kctxzn

#for detailed comments see subroutine 'marginalx'
@njit
def marginalz(zt,M,stepmags,K): #given zt, compute marginal through RR method
    steps=int(stepmags[-1]) #maximum number of steps
    xytrajs=np.zeros((K,2,2))
    Uk=np.zeros(K)
    Ukt=Uk
    Umin=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0
    
    Kctxyn=0

    xn=np.zeros((K,2))
    yn=np.zeros((K,2))

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
            
        #propagate ensemble of y trajectories for one step and accumulate weight
        for k in range(K):
            xytrajs[k,:,0]=xytrajs[k,:,1]
            xn[k,:],yn[k,:]=propagatexy(xytrajs[k,0,0],xytrajs[k,1,0],zt[i:i+2],1)
            xytrajs[k,0,1]=xn[k,-1]
            xytrajs[k,1,1]=yn[k,-1]
            Uj=-trajprob(xytrajs[k,0,:],xytrajs[k,1,:],zt[i:i+2],1,np.array([1]))[-1,-1]
            U0=-trajprob0xy(xytrajs[k,0,:],xytrajs[k,1,:],zt[i:i+2],1,np.array([1]))[-1]
            Uk[k]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Mcount=Mcount+1

    return wval,Kctxyn

#for detailed comments see subroutine 'marginalx'
@njit
def marginalxy(xt,yt,M,stepmags,K): #given xt and yt, compute marginal through Rosenbluth-Rosenbluth method
    steps=int(stepmags[-1]) #maximum number of steps
    ztrajs=np.zeros((K,2))
    Uk=np.zeros(K)
    Ukt=Uk
    Uktold=np.zeros(K)
    Umin=0.0
    #Ufullmin=0.0
    Uminold=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0

    Kctzn=0

    Tvals=np.zeros((M,2)) #cumulative contributions to the numerator in TE expression in both directions
    Tstepwise=np.zeros(2) #changes in Tvals at every step
    probs=np.zeros((K,1,4))
    #Ufull=np.zeros((K,2))

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
            #Ufull[:,0]=(Ufull[:,0])[zindices]

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.

        #propagate ensemble of x trajectories for one step and accumulate weight
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            ztrajs[k,0]=ztrajs[k,1]
            ztrajs[k,1]=propagatez(xt[i:i+2],yt[i:i+2],ztrajs[k,0],1)[-1]
            probs[k,:,:]=trajprob(xt[i:i+2],yt[i:i+2],ztrajs[k,:],1,np.ones(1))
            Uj=-probs[k,-1,-1]
            U0=-trajprob0z(xt[i:i+2],yt[i:i+2],ztrajs[k,:],1,np.ones(1))[-1]
            Uk[k]+=Uj-U0
            #Ufull[k,1]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #contribution to transfer entropy
        #need to keep weights of each trajectory from start, not just the averaged ones like wval
        #during resampling, keep track of individual weights, but upto the previous step
        #Ufullmin=np.amin(Ufull[:,0])
        Tstepwise[0]+=np.log(np.dot(np.exp(probs[:,-1,0]),np.exp(-Uktold))/K)-Uminold
        Tstepwise[1]+=np.log(np.dot(np.exp(probs[:,-1,1]),np.exp(-Uktold))/K)-Uminold
        Tstepwise[0]-=np.log(np.sum(np.exp(-Uktold))/K)-Uminold
        Tstepwise[1]-=np.log(np.sum(np.exp(-Uktold))/K)-Uminold
        #Ufull[:,0]=Ufull[:,1]

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount,:]=Tstepwise
            Mcount=Mcount+1

    return wval,Kctzn,Tvals
    
#for detailed comments see subroutine 'marginalx'
@njit
def marginalyz(yt,zt,M,stepmags,K): #given yt and zt, compute marginal through Rosenbluth-Rosenbluth method
    steps=int(stepmags[-1]) #maximum number of steps
    xtrajs=np.zeros((K,2))
    Uk=np.zeros(K)
    Ukt=Uk
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    #Ufullmin=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0

    Kctxn=0

    Tvals=np.zeros((M,2)) #cumulative contributions to the numerator in TE expression in both directions
    Tstepwise=np.zeros(2) #changes in Tvals at every step
    probs=np.zeros((K,1,4))
    #Ufull=np.zeros((K,2))

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
            #Ufull[:,0]=(Ufull[:,0])[xindices]

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.

        #propagate ensemble of x trajectories for one step and accumulate weight
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            xtrajs[k,0]=xtrajs[k,1]
            xtrajs[k,1]=propagatex(xtrajs[k,0],yt[i:i+2],zt[i:i+2],1)[-1]
            probs[k,:,:]=trajprob(xtrajs[k,:],yt[i:i+2],zt[i:i+2],1,np.ones(1))
            Uj=-probs[k,-1,-1]
            U0=-trajprob0x(xtrajs[k,:],yt[i:i+2],zt[i:i+2],1,np.ones(1))[-1]
            Uk[k]+=Uj-U0
            #Ufull[k,1]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #Ufullmin=np.amin(Ufull[:,0]) 
        Tstepwise[0]+=np.log(np.dot(np.exp(probs[:,-1,1]),np.exp(-Uktold))/K)-Uminold
        Tstepwise[1]+=np.log(np.dot(np.exp(probs[:,-1,2]),np.exp(-Uktold))/K)-Uminold
        Tstepwise[0]-=np.log(np.sum(np.exp(-Uktold))/K)-Uminold
        Tstepwise[1]-=np.log(np.sum(np.exp(-Uktold))/K)-Uminold
        #Ufull[:,0]=Ufull[:,1]

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount,:]=Tstepwise
            Mcount=Mcount+1

    return wval,Kctxn,Tvals

#for detailed comments see subroutine 'marginalx'
@njit
def marginalxz(xt,zt,M,stepmags,K): #given xt and zt, compute marginal through Rosenbluth-Rosenbluth method
    steps=int(stepmags[-1]) #maximum number of steps
    ytrajs=np.zeros((K,2))
    Uk=np.zeros(K)
    Ukt=Uk
    Uktold=np.zeros(K)
    Umin=0.0
    Uminold=0.0
    #Ufullmin=0.0
    wval=np.zeros(M) #marginal probability only upto the given duration
    Keff=K
    w=K
    Mcount=0

    Kctyn=0

    Tvals=np.zeros((M,2)) #cumulative contributions to the numerator in TE expression in both directions
    Tstepwise=np.zeros(2) #changes in Tvals at every step
    probs=np.zeros((K,1,4))
    #Ufull=np.zeros((K,2))

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
            #Ufull[:,0]=(Ufull[:,0])[yindices]

            #reset Uk
            Uk[:]=0.
            Ukt[:]=0.
            Umin=0.

        #propagate ensemble of x trajectories for one step and accumulate weight
        Uktold=Ukt
        Uminold=Umin
        for k in range(K):
            ytrajs[k,0]=ytrajs[k,1]
            ytrajs[k,1]=propagatey(xt[i:i+2],ytrajs[k,0],zt[i:i+2],1)[-1]
            probs[k,:,:]=trajprob(xt[i:i+2],ytrajs[k,:],zt[i:i+2],1,np.ones(1))
            Uj=-probs[k,-1,-1]
            U0=-trajprob0y(xt[i:i+2],ytrajs[k,:],zt[i:i+2],1,np.ones(1))[-1]
            Uk[k]+=Uj-U0
            #Ufull[k,1]+=Uj-U0

        Umin=np.amin(Uk)
        Ukt=Uk-Umin
        w=np.sum(np.exp(-Ukt)) #the subtraction of Umin keeps the magnitude of the exponentials within machine capacity

        #Ufullmin=np.amin(Ufull[:,0]) 
        Tstepwise[0]+=np.log(np.dot(np.exp(probs[:,-1,0]),np.exp(-Uktold))/K)-Uminold
        Tstepwise[1]+=np.log(np.dot(np.exp(probs[:,-1,2]),np.exp(-Uktold))/K)-Uminold
        Tstepwise[0]-=np.log(np.sum(np.exp(-Uktold))/K)-Uminold
        Tstepwise[1]-=np.log(np.sum(np.exp(-Uktold))/K)-Uminold
        #Ufull[:,0]=Ufull[:,1]

        #recalculate Keff
        Keff=w**2/np.sum(np.exp(-Ukt)**2)

        #check if this is the last loop for any trajectory to account for remaining weight
        if i==stepmags[Mcount]-1:
            wval[Mcount]+=np.log(w/K)-Umin
            Tvals[Mcount,:]=Tstepwise
            Mcount=Mcount+1

    return wval,Kctyn,Tvals

########this is the master subroutine for sampling the logarithm in the TE expression for different pairs of trajectories. This subroutine calls marginalization subroutines.
@njit(parallel=True)
def rrpws(s):
    M=np.shape(stepmags)[0] #calculate transfer entropies for all timesteps 

    Iestxz=np.zeros((N,M)) #holder for mutual information
    Kctxyd=np.zeros(N) #holder for the number of pruning/enrichment, purely for studying the efficiency of the algorithm 
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

    xts=np.zeros((N,stepmags[-1]+1)) #holder for outer trajectories
    yts=np.zeros((N,stepmags[-1]+1))
    zts=np.zeros((N,stepmags[-1]+1))

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
        xts[i,:],yts[i,:],zts[i,:]=propagatexyz(xyzN[i,0],xyzN[i,1],xyzN[i,2],stepmags[-1])

    for i in prange(N): #prange explicitly parallelizes with numba

	#comment out irrelevant lines if transfer entropy in every possible combination is not needed
        pxz[i,:],Kctyd[i],Tvalsxz[i,:,:]=marginalxz(xts[i,:],zts[i,:],M,stepmags,K) #probability, resampling number, numerator in TE expression
        Iestxz[i,:]+=pxz[i,:] #numerator in MI expression
        pyz[i,:],Kctxd[i],Tvalsyz[i,:,:]=marginalyz(yts[i,:],zts[i,:],M,stepmags,K) #probability, resampling number, numerator in TE expression
        Iestyz[i,:]+=pyz[i,:] #numerator in MI expression
        pxy[i,:],Kctzd[i],Tvalsxy[i,:,:]=marginalxy(xts[i,:],yts[i,:],M,stepmags,K) #probability, resampling number, numerator in TE expression
        Iestxy[i,:]+=pxy[i,:] #numerator in MI expression

        px[i,:],Kctyzd[i]=marginalx(xts[i,:],M,stepmags,K) 
        py[i,:],Kctxzd[i]=marginaly(yts[i,:],M,stepmags,K)
        pz[i,:],Kctxyd[i]=marginalz(zts[i,:],M,stepmags,K) 

        Iestxz[i,:]-=px[i,:]+pz[i,:] #MI denominator
        Tvalsxz[i,:,0]-=px[i,:] #TE denominator
        Tvalsxz[i,:,1]-=pz[i,:] #TE denominator
        Iestxy[i,:]-=px[i,:]+py[i,:] #MI denominator
        Tvalsxy[i,:,0]-=px[i,:] #TE denominator
        Tvalsxy[i,:,1]-=py[i,:] #TE denominator
        Iestyz[i,:]-=py[i,:]+pz[i,:] #MI denominator
        Tvalsyz[i,:,0]-=py[i,:] #TE denominator
        Tvalsyz[i,:,1]-=pz[i,:] #TE denominator

    return Iestxy,Iestyz,Iestxz,Kctxd,Kctyd,Kctzd,Kctxyd,Kctyzd,Kctxzd,Tvalsxy,Tvalsyz,Tvalsxz

#s=7
Iestxy,Iestyz,Iestxz,Kctxd,Kctyd,Kctzd,Kctxyd,Kctyzd,Kctxzd,Tvalsxy,Tvalsyz,Tvalsxz=rrpws(s)
Kct=np.stack((Kctxd,Kctyd,Kctzd,Kctxyd,Kctyzd,Kctxzd),axis=-1)

#Monte-Carlo average outside the logarithm in the TE expression
iest2xy=np.mean(Iestxy,axis=0)
iest2yz=np.mean(Iestyz,axis=0)
iest2xz=np.mean(Iestxz,axis=0)
tvals2xy=np.mean(Tvalsxy,axis=0)
tvals2yz=np.mean(Tvalsyz,axis=0)
tvals2xz=np.mean(Tvalsxz,axis=0)

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


