"""

"""

import numpy as np

def calc_cumulants(image,mask):
    """
    """
    numch = image.shape[0]
    M     = mask.sum()
    mask4d = np.repeat(mask[np.newaxis,:,:,:],image.shape[0],axis=0)

    kappa1 = np.array([ch[:,:-1][mask].mean() for ch in image])
    kappa2 = np.zeros([numch,numch])*np.nan
    mu22   = np.zeros([numch,numch])*np.nan

    imagedelta1   = image[:,:,:-1][mask4d].reshape(numch,M) - kappa1[:,None]
    imagedelta2   = image[:,:,1:][mask4d].reshape(numch,M) - kappa1[:,None]
    imagedelta1sq = imagedelta1**2
    imagedelta2sq = imagedelta2**2

    for ch1 in range(numch):
        for ch2 in range(ch1,numch):
            kappa2[ch1,ch2] = np.mean(imagedelta1[ch1] * imagedelta2[ch2])
            mu22[ch1,ch2]   = np.mean(imagedelta1sq[ch1] * imagedelta2sq[ch2])
        
    return kappa1, kappa2, mu22


def calc_cumulants_fr(image, mask, maskwide):
    """
    """
    numch = image.shape[0]
    
    kappa2 = np.zeros([numch,numch])*np.nan
    mu22 = np.zeros([numch,numch])*np.nan
    
    kappa1 = np.array([ np.array([ch[:,:,t][maskwide[:,:,t]].mean() for t in range(image.shape[-1])]) for ch in image])
    
    imagedelta = image - kappa1[:,None,None,:]
    imagedeltasq = imagedelta**2
    
    for ch1 in range(numch):
        for ch2 in range(ch1,numch):
            kappa2[ch1,ch2] = np.mean(imagedelta[ch1,:,:-1][mask] * imagedelta[ch2,:,1:][mask])
            mu22[ch1,ch2]   = np.mean(imagedeltasq[ch1,:,:-1][mask] * imagedeltasq[ch2,:,1:][mask])
    
    pixels = mask.sum(axis=(0,1)).astype('float')        
    kappa1 = (kappa1*pixels).sum(axis=1) / pixels.sum()
    
    return kappa1, kappa2, mu22


def calc_variance(kappa1,kappa2,mu22,M):
    """
    """
    numch = len(kappa1)
    varkappa1 = (kappa2[range(numch),range(numch)] + kappa1)/M
    varkappa2 = (mu22-kappa2**2)/M
    
    return varkappa1, varkappa2


def condense(data,config):
    """
    Condenses information (e.g. emission spectrum) at higher spectral resolution into fewer channels specified by 
    config.
    """
    newdata=np.zeros(len(config))
    for i in range(len(config)):
        newdata[i]=data[config[i]].sum()
        
    return newdata


def collapse(data,channelidx):
    """
    Combines data simulated at highest spectral resolution into fewer channels specified by config.
    """
    newdata = np.zeros([len(channelidx),data.shape[1],data.shape[2],data.shape[3]])
    for i in range(len(channelidx)):
        newdata[i] = data[channelidx[i]].sum(axis=0)
        
    return newdata