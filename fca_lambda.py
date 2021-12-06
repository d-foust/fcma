"""
Created on Fri Mar 09 13:18:53 2018

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report
from fca import get_var_cumulants
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def get_cumulants_lambda(N,eps,B,gamma,k0=None,Pa=None,fdead=None):
    """
    N: emitter number
    eps: row for each species (matches length of N), column for each fluorophore (matches rows in B)
    B: row for each fluorophore, column for each detector element
    gamma: shape factors
    k0: background count rates
    Pa: afterpulsing probabilities
    fdead: fraction of sample period spent in 
    """
    numcomp=len(N)      # number of emitter species
    numcol=B.shape[1]   # number of detector elements
    
    kappa1=np.zeros(numcol)
    kappa2=np.zeros([numcol,numcol])
    kappa12=np.zeros([numcol,numcol])
    kappa21=np.zeros([numcol,numcol])
    kappa22=np.zeros([numcol,numcol])
    for comp in range(numcomp):
        epslambda=np.dot(eps[comp,:],B)
        kappa1+=N[comp]*epslambda
        kappa2[range(numcol),range(numcol)]  += N[comp] * epslambda**2 * gamma[1]
        for col in range(1,numcol):
            kappa2[range(0,numcol-col),range(col,numcol)]  += N[comp]*epslambda[:numcol-col]*epslambda[col:]*gamma[1]
            kappa2[range(col,numcol),range(0,numcol-col)]  += N[comp]*epslambda[:numcol-col]*epslambda[col:]*gamma[1]
            kappa12[range(0,numcol-col),range(col,numcol)] += N[comp]*epslambda[:numcol-col]*epslambda[col:]**2*gamma[2]
            kappa21[range(0,numcol-col),range(col,numcol)] += N[comp]*epslambda[:numcol-col]**2*epslambda[col:]*gamma[2]
            kappa22[range(0,numcol-col),range(col,numcol)] += N[comp]*epslambda[:numcol-col]**2*epslambda[col:]**2*gamma[3]
            
    if k0 is not None:
        kappa1+=k0      
    if Pa is not None and Pa.sum()>0:
        kappa1_new = np.copy(kappa1)
        kappa2_new = np.copy(kappa2)
        kappa1_new *= (1+Pa)
        kappa2_new[range(numcol),range(numcol)] = kappa2[range(numcol),range(numcol)]*(1+Pa**2)+2*Pa*(kappa1+kappa2[range(numcol),range(numcol)])
        for col1 in range(numcol-1):
            for col2 in range(col1+1,numcol):
                kappa2_new[col1,col2] *= (1+Pa[col1])*(1+Pa[col2])
        kappa1=kappa1_new
        kappa2=kappa2_new
            
    # set empties to nan
#     x,y=np.meshgrid(range(numcol),range(numcol))
#     kappa2[y>x]=np.nan
#     kappa12[y>x]=np.nan
#     kappa21[y>x]=np.nan
#     kappa22[y>x]=np.nan
                
    return kappa1,kappa2,kappa12,kappa21,kappa22
    
def fit_cumulants_lambda(N0,eps0,B,kappa1,varkappa1,kappa2,varkappa2,gamma,k0=None,Pa=None,Nvary=None,epsvary=None,gammavary=None,Pavary=None,fdead=None,constr=None):
    """
    """
    if Nvary is None: Nvary=np.ones(len(N0))
    if epsvary is None: epsvary=np.ones(eps0.shape)
    if gammavary is None: gammavary=np.zeros(4)
    if Pa is None: Pa=np.zeros(len(kappa1))
    if Pavary is None: Pavary=np.zeros(len(kappa1))

    pars=Parameters()
    numcomp=len(N0)
    numcol=B.shape[1] # number of detection elements
    numfl=B.shape[0]
    for comp in range(numcomp):
        pars.add('N'+str(comp+1),value=N0[comp,1],min=N0[comp,0],max=N0[comp,2],vary=Nvary[comp])
#         pars.add('N'+str(comp+1),value=N0[comp],min=N0[comp],max=N0[comp],vary=Nvary[comp])
        for fl in range(numfl):
            pars.add('eps'+str(comp+1)+chr(65+fl),value=eps0[comp,fl],min=0,max=100,vary=epsvary[comp,fl],expr=constr[comp][fl])
    for col in range(numcol):
        pars.add('Pa'+chr(65+col%26)*(1+int(col/26)),value=Pa[col],min=0,max=1,vary=Pavary[col])
    for order in range(4):
        pars.add('gamma'+str(order+1),value=gamma[order],min=0,max=1,vary=gammavary[order])
        
    return minimize(res_cumulants_lambda,pars,args=(kappa1,varkappa1,kappa2,varkappa2,B,k0,fdead,numcomp))
        
#            
def res_cumulants_lambda(pars,kappa1,varkappa1,kappa2,varkappa2,B,k0,fdead,numcomp):
    """
    """
    vals = pars.valuesdict()
    numcol=B.shape[1] # number of detection elements
    numfl=B.shape[0] # number of fluorophores in sample
    N = np.zeros(numcomp)
    eps = np.zeros([numcomp,numfl])
    Pa = np.zeros(numcol)
    gamma=np.zeros(4)
    for comp in range(numcomp):
        N[comp] = vals['N'+str(comp+1)]
        for fl in range(numfl):
            eps[comp,fl]=vals['eps'+str(comp+1)+chr(fl+65)]
    for col in range(numcol):
        Pa[col]=vals['Pa'+chr(65+col%26)*(1+int(col/26))]
    for order in range(4):
        gamma[order]=vals['gamma'+str(order+1)]
        
    kappa1fit,kappa2fit,kappa12fit,kappa21fit,kappa22fit = get_cumulants_lambda(N,eps,B,gamma,k0,Pa,fdead)
    res1 = (kappa1-kappa1fit)/np.sqrt(varkappa1)
    res2 = (kappa2-kappa2fit)/np.sqrt(varkappa2)
    res1 = res1[~np.isnan(res1)]
    res2 = res2[~np.isnan(res2)]
    res = np.append(res1,res2)
    return res
    
def retrieve_cumulant_fit_lambda(fit,numcomp,numcol,numfl):
    """
    """
    # initial storage space
    vals=fit.params
    N=np.zeros(numcomp); Nerr=np.zeros(numcomp)
    eps=np.zeros([numcomp,numfl]); epserr=np.zeros([numcomp,numfl])
    Pa=np.zeros(numcol); Paerr=np.zeros(numcol)
    gamma=np.zeros(4); gammaerr=np.zeros(4)
    # retrieve fit number and brightness
    for comp in range(numcomp):
        N[comp]=vals['N'+str(comp+1)].value
        Nerr[comp]=vals['N'+str(comp+1)].stderr
        for fl in range(numfl):
            eps[comp,fl]=vals['eps'+str(comp+1)+chr(65+fl)].value
            epserr[comp,fl]=vals['eps'+str(comp+1)+chr(65+fl)].stderr
    # retrieve fit afterpulsing probabilities
    for col in range(numcol):
        Pa[col]=vals['Pa'+chr(65+col%26)*(1+int(col/26))].value
        Paerr[col]=vals['Pa'+chr(65+col%26)*(1+int(col/26))].stderr
    # retrieve fit for shape factors
    for order in range(4):
        gamma[order]=vals['gamma'+str(order+1)].value
        gammaerr[order]=vals['gamma'+str(order+1)].stderr
    # retrieve goodness of fit data
    residuals=fit.residual
    chi2=fit.redchi
    return N,Nerr,eps,epserr,Pa,Paerr,gamma,gammaerr,residuals,chi2
    
def plot_lfca_fit(kappa1exp,kappa2exp,res,fit,Blambda,numcomp,lam,colrs,k0=None):
    
    numcol=len(kappa1exp)
    numfl=Blambda.shape[0]
    
    Nfit,Nerr,epsfit,epserr,Pa,Paerr,gamma,gammaerr,residuals,chi2=retrieve_cumulant_fit_lambda(fit,numcomp,numcol,numfl)
    kappa1fit,kappa2fit,kappa12,kappa21,kappa22=get_cumulants_lambda(Nfit,epsfit,Blambda,gamma,k0=k0)
    kappa1comp=np.zeros([numcomp,len(kappa1exp)])
    order=Nfit.argsort()
    for comp in range(numcomp):
        kappa1_,a,b,c,d=get_cumulants_lambda(np.array([Nfit[comp]]),np.array([epsfit[comp]]),Blambda,gamma)
        kappa1comp[comp,:]=kappa1_
    kappa1comp=kappa1comp[order]
    colrs = [colrs[i] for i in order]
    
    plt.figure(figsize=(16,13))

    plt.subplot(221)
    plt.title('2nd Order Experiment',fontsize=24)
    plt.imshow(kappa2exp,interpolation='none',cmap='inferno',clim=(0,np.nanmax(kappa2fit)),norm=colors.PowerNorm(gamma=.2))
    plt.gca().invert_yaxis()
    cb=plt.colorbar(ticks=[np.nanmax(kappa2fit)*1e-3,np.nanmax(kappa2fit)*1e-2,np.nanmax(kappa2fit)*1e-1,np.nanmax(kappa2fit)],format='%.0e')
    cb.ax.tick_params(labelsize=14)
    plt.xticks(range(0,len(kappa1exp),2),lam[0::2],rotation='vertical',fontsize=16)
    plt.yticks(range(0,len(kappa1exp),2),lam[0::2],fontsize=16)
    plt.xlabel(r'Channel $\mathit{i}$',fontsize=25)
    plt.ylabel(r'Channel $\mathit{j}$',fontsize=25)
    
    plt.subplot(222)
    plt.title('2nd Order Fit',fontsize=24)
    plt.imshow(kappa2fit,interpolation='none',cmap='inferno',clim=(0,np.nanmax(kappa2fit)),norm=colors.PowerNorm(gamma=.2))
    plt.gca().invert_yaxis()
    cb=plt.colorbar(ticks=[np.nanmax(kappa2fit)*1e-3,np.nanmax(kappa2fit)*1e-2,np.nanmax(kappa2fit)*1e-1,np.nanmax(kappa2fit)],format='%.0e')
    cb.ax.tick_params(labelsize=14)
    plt.xticks(range(0,len(kappa1exp),2),lam[0::2],rotation='vertical',fontsize=16)
    plt.yticks(range(0,len(kappa1exp),2),lam[0::2],fontsize=16)
    plt.xlabel(r'Channel $\mathit{i}$',fontsize=25)
    plt.ylabel(r'Channel $\mathit{j}$',fontsize=25)
    
    plt.subplot(223)
    h=np.nanmax(res)
    l=np.nanmin(res)
    bound=np.abs([h,l]).max()
    plt.title('2nd Order Residuals',fontsize=24)
    plt.imshow(res,interpolation='none',cmap='bwr',clim=(-bound,bound))
    plt.gca().invert_yaxis()
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xticks(range(0,len(kappa1exp),2),lam[0::2],rotation='vertical',fontsize=16)
    plt.yticks(range(0,len(kappa1exp),2),lam[0::2],fontsize=16)
    plt.xlabel(r'Channel $\mathit{i}$',fontsize=25)
    plt.ylabel(r'Channel $\mathit{j}$',fontsize=25)
    
    lameven=np.arange(lam[0],lam[0]+8.9*(len(kappa1exp)),8.9)
    plt.subplot(224)
    plt.title('1st Order',fontsize=24)
    for comp in range(numcomp):
        if Nfit[order[comp]]<0:
            plt.bar(lameven,kappa1comp[comp],width=8.9,align='center',bottom=kappa1comp[0:comp,:].sum(axis=0),color='none',edgecolor=colrs[comp],linewidth=2,zorder=10)
        elif Nfit[order[comp]]>=0:
            plt.bar(lameven,kappa1comp[comp],width=8.9,align='center',bottom=kappa1comp[0:comp,:].sum(axis=0),color=colrs[comp],edgecolor='none')
#    plt.bar(lameven,k0,width=8.9,align='center',bottom=kappa1comp.sum(axis=0),color='#929591',edgecolor='none')
    
    plt.plot(lameven,kappa1exp,'kx',ms=8,mew=2,label='Experiment')
    plt.xticks(lam[0::2],rotation='vertical',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([lameven[0]-6,lameven[-1]+6])
#    plt.ylim([-.05,2.5])
#    plt.xlabel(r'$\lambda$',fontsize=27)
    plt.xlabel(r'Channel $\mathit{i}$',fontsize=26)
    plt.ylabel(r'$\kappa _{[1]}(\mathit{i})$',fontsize=26)
    plt.tight_layout()