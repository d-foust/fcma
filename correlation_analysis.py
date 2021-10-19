from numpy.ma import average
from numpy import ones, conj, empty, nan, abs, exp, isnan
import numpy as np
from numpy.fft import rfftn, irfftn, fftshift, rfft2, irfft2
from itertools import combinations_with_replacement
from lmfit import Parameters, minimize

def calc_corr_fxns(image, mask):
    """
    New version 6/23/21
    """
    nframes, nchannels, nrows, ncols = image.shape
    channel_pairs = list(combinations_with_replacement(range(nchannels),2))
    
    image_delta = np.zeros([nchannels, nrows, ncols])
    image_delta_sq = np.zeros([nchannels, nrows, ncols])
    image_ft    = np.zeros([nchannels, nrows, ncols//2+1], dtype='complex')
    image_sq_ft = np.zeros([nchannels, nrows, ncols//2+1], dtype='complex')
    corr_fxns   = np.zeros([len(channel_pairs), nrows, ncols])
    mu11        = np.zeros([len(channel_pairs), nrows, ncols])
    mu22        = np.zeros([len(channel_pairs), nrows, ncols])
    var         = np.zeros([len(channel_pairs), nrows, ncols])
    mask_ft     = np.zeros([nrows, ncols//2+1])
    R           = np.zeros([nrows, ncols])
    
    for frame in range(nframes):
        mask_ft = rfft2(mask[frame])
        R += irfft2(mask_ft * np.conj(mask_ft))
        
        for channel in range(nchannels):
            image_delta[channel] = image[frame,channel] - image[frame,channel][mask[frame]].mean()
            image_delta[channel][~mask[frame]] = 0
            image_delta_sq[channel] = image_delta[channel]**2
            image_ft[channel] = rfft2(image_delta[channel])
            image_sq_ft[channel] = rfft2(image_delta_sq[channel])
            
        for ich, (ch1, ch2) in enumerate(channel_pairs):
            corr_fxns[ich] += irfft2(image_ft[ch1] * np.conj(image_ft[ch2]))
            mu22[ich]      += irfft2(image_sq_ft[ch1] * np.conj(image_sq_ft[ch2]))
            
    I_av = np.zeros(nchannels)
    for channel in range(nchannels):
        I_av[channel] = image[:,channel][mask].mean()
            
    for ich, (ch1, ch2) in enumerate(channel_pairs):
        mu11[ich] = corr_fxns[ich] / R
        mu22[ich] /= R
        var[ich] = (mu11[ich]**2 - 2*mu11[ich] + mu22[ich]) / (R * I_av[ch1]**2 * I_av[ch2]**2)
        corr_fxns[ich] /= (R * I_av[ch1] * I_av[ch2])
        
    corr_fxns[:,0,0] = np.nan
    corr_fxns[:] = fftshift(corr_fxns, axes=(1,2))
    
    var[:,0,0] = np.nan
    var[:] = fftshift(var, axes=(1,2))
            
    return corr_fxns, var**0.5

def calc_corr_fxns_old(image, roi):
    numframes, numchannels = image.shape[:2]
    numrows, numcols = image.shape[2:]
    roi_ext = roi[:,None] * ones([1,numchannels,1,1], dtype='bool')
    averages = average(image, weights=roi_ext, axis=(2,3))
    image_delta = image - averages[:,:,None,None]
    image_delta[~roi_ext] = 0
    image_delta_sq = image_delta**2
    
    roi_ft = rfftn(roi)
    R = irfftn(roi_ft * conj(roi_ft))[0]
    
    image_ft = rfftn(image_delta, axes=(0,2,3))
    image_sq_ft = rfftn(image_delta_sq, axes=(0,2,3))
    
    ch_pairs = list(combinations_with_replacement(range(numchannels),2))
    numcorrfxns = len(ch_pairs)
    mu11 = empty([numcorrfxns, numrows, numcols])
    mu22 = empty([numcorrfxns, numrows, numcols])
    
    channel_averages = average(image, weights=roi_ext, axis=(0,2,3))
    
    for ich, (ch1,ch2) in enumerate(ch_pairs):
        mu11[ich] = irfftn(image_ft[:,ch1] * conj(image_ft[:,ch2]))[0] / R
        mu22[ich] = irfftn(image_sq_ft[:,ch1] * conj(image_sq_ft[:,ch2]))[0] / R
        
    mu11[:,0,0] = nan; mu22[:,0,0] = nan
    
    var = (mu11**2 - 2*mu11 + mu22) / (R[None,:,:] * ones([numcorrfxns,1,1]))
    mu11 = fftshift(mu11, axes=(1,2)) 
    var = fftshift(var, axes=(1,2))
    
    cfs = empty(mu11.shape)
    sigma = empty(var.shape)
    
    for ich, (ch1, ch2) in enumerate(ch_pairs):
        cfs[ich] = mu11[ich] / (channel_averages[ch1] * channel_averages[ch2])
        sigma[ich] = var[ich]**0.5 / (channel_averages[ch1] * channel_averages[ch2])
        
    return cfs, sigma

def G_diff(xi, phi, tau_px, tau_ln, w0, S, n_photon, D, xi0):
    term1 = (1 + (4*n_photon*D)*abs(tau_px*(xi-xi0) + tau_ln*phi) / w0**2)**-1
    term2 = (1 + (4*n_photon*D)*abs(tau_px*(xi-xi0) + tau_ln*phi) / (S*w0)**2)**-0.5
    return term1 * term2

def G_space(xi, phi, tau_px, tau_ln, dr, w0, n_photon, D, xi0):
    return exp(-dr**2 * ((xi-xi0)**2 + phi**2) / (w0**2 + 4*n_photon*D*abs(tau_px*(xi-xi0) + tau_ln*phi)))

def G_mob(xi, phi, tau_px, tau_ln, dr, w0, S, n_photon, G0, D, xi0):
    """
    xi: spatial lag in scanning direction (pixels)
    phi: spatial lag in orthogonal direction (pixels)
    tau_px: pixel dwell time (s)
    tau_ln: lag time between scanned lines (s)
    dr: distance between pixels (um)
    w0: lateral width of detection volume (um)
    S: ration of axial width of detection volume to lateral width (unitless [um/um])
    n_photon: number of photons used for excitation (e.g. 2-photon -> 2)
    G0: amplitude of correlation function (unitless)
    D: diffusion coefficient (um^2/s)
    xi0: shift along scanning direction (pixels)
    """
    return G0 * G_diff(xi, phi, tau_px, tau_ln, w0, S, n_photon, D, xi0) \
              * G_space(xi, phi, tau_px, tau_ln, dr, w0, n_photon, D, xi0)

def fit_G_mob(G_exp, G_exp_sigma, xi, phi, tau_px, tau_ln, dr, w0, S, n_photon, G0_init, D_init, xi0_init):
    pars = Parameters()
    pars.add('G0', min=G0_init[0], value=G0_init[1], max=G0_init[2])
    pars.add('D', min=D_init[0], value=D_init[1], max=D_init[2])
    pars.add('xi0', min=xi0_init[0], value=xi0_init[1], max=xi0_init[2])
    
    return minimize(res_G_mob, pars, 
                    args=(G_exp, G_exp_sigma, xi, phi, tau_px, tau_ln, dr, w0, S, n_photon))
    
def res_G_mob(pars, G_exp, G_exp_sigma, xi, phi, tau_px, tau_ln, dr, w0, S, n_photon):
    vals = pars.valuesdict()
    G0 = vals['G0']
    D  = vals['D']
    xi0 = vals['xi0']
    
    G_fit = G_mob(xi, phi, tau_px, tau_ln, dr, w0, S, n_photon, G0, D, xi0)
    residuals = (G_exp - G_fit) / G_exp_sigma
    residuals = residuals[~isnan(residuals)]
    
    return residuals   

def G_diff_2D(xi, phi, tau_px, tau_ln, w0, S, n_photon, D, xi0):
    """
    Diffusion related component of 2D model correlation function.
    """
#     term1 = (1 + (4*n_photon*D)*abs(tau_px*(xi-xi0) + tau_ln*phi) / w0**2)**-0.5
    term1 = (1 + (4*n_photon*D)*abs(tau_px*(xi-xi0) + tau_ln*phi) / w0**2)**-1
    term2 = 1

    return term1 * term2  

def G_mob_2D(xi, phi, tau_px, tau_ln, dr, w0, S, n_photon, G0, D, xi0):
    return G0 * G_diff_2D(xi, phi, tau_px, tau_ln, w0, S, n_photon, D, xi0) \
              * G_space(xi, phi, tau_px, tau_ln, dr, w0, n_photon, D, xi0)

def fit_G_mob_2D(G_exp, G_exp_sigma, xi, phi, tau_px, tau_ln, dr, w0, S, n_photon, G0_init, D_init, xi0_init):
    pars = Parameters()
    pars.add('G0', min=G0_init[0], value=G0_init[1], max=G0_init[2])
    pars.add('D', min=D_init[0], value=D_init[1], max=D_init[2])
    if xi0_init == 0:
        pars.add('xi0', value=0, vary=False)
    else:
        pars.add('xi0', min=xi0_init[0], value=xi0_init[1], max=xi0_init[2], vary=True)
    
    return minimize(res_G_mob_2D, pars, 
                    args=(G_exp, G_exp_sigma, xi, phi, tau_px, tau_ln, dr, w0, S, n_photon))

def res_G_mob_2D(pars, G_exp, G_exp_sigma, xi, phi, tau_px, tau_ln, dr, w0, S, n_photon):
    """
    """
    vals = pars.valuesdict()
    G0 = vals['G0']
    D  = vals['D']
    xi0 = vals['xi0']
    
    G_fit = G_mob_2D(xi, phi, tau_px, tau_ln, dr, w0, S, n_photon, G0, D, xi0)
    residuals = (G_exp - G_fit) / G_exp_sigma
    residuals = residuals[~isnan(residuals)]
    
    return residuals   