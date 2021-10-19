"""
Functions for calculating spectrally unmixed images using the method described in: 
Schrimpf, W., V. Lemmens, N. Smisdom, M. Ameloot, D.C. Lamb, and J. Hendrix. 2018. Crosstalk-free multicolor RICS using spectral weighting. Methods. 140–141:97–111.
"""

from numpy import zeros, dot, newaxis, average, ones
from numpy.linalg import inv, multi_dot

def decompose(image, reference_spectra):
    """
    Spectral decomposition (unmixing) of image given provided reference spectra.
    image: (nframes, nchannels, nrows, ncols)
    reference_spectra: (nspecies, nchannels)
    """
    numframes = image.shape[0]
    numchannels = image.shape[1]
    numspecies = reference_spectra.shape[0]
    
    image_decomp = zeros([numframes, numspecies, image.shape[2], image.shape[3]])
    
    # unmix frame by frame
    for frame in range(numframes):
        D = zeros([numchannels, numchannels])
        D[range(numchannels),range(numchannels)] = image[frame].mean(axis=(1,2))**-1
        M = reference_spectra
        weights = dot(inv(dot(M ,dot(D, M.T))), dot(M,D))
        image_decomp[frame] = (weights[...,newaxis,newaxis] * image[frame]).sum(axis=1)
        
    return image_decomp

def decompose_with_mask(image, reference_spectra, mask):
    """
    Spectral decomposition where 
    < worried more about photobleaching, frame-to-frame variation than uncertainty >
    """
    numframes = image.shape[0]
    numchannels = image.shape[1]
    numspecies = reference_spectra.shape[0]
    
    image_decomp = zeros([image.shape[0], numspecies, image.shape[2], image.shape[3]])
    for frame in range(numframes):
        D = zeros([numchannels, numchannels])
        D[range(numchannels),range(numchannels)] = average(image[frame],
                                                    weights=mask[frame][None,:,:]*ones([numchannels,1,1],dtype='bool'),
                                                    axis=(1,2))**-1
        M = reference_spectra
        weights = dot(inv(dot(M ,dot(D, M.T))), dot(M,D))
        image_decomp[frame] = (weights[...,newaxis,newaxis] * image[frame]).sum(axis=1)
        
    return image_decomp

def decompose_with_mask_all(image, reference_spectra, mask):
    """
    Spectral decomposition where single composite spectrum is determined from all frames within mask.
    < more worried about frame-to-frame uncertainty, less worried about photobleaching + other changes >
    image: dimensions (time, color, x, y)
    reference spectra: matrix with dimensions (# species) x (# detection channels); should be normalized to unity 
                        i.e. sum along rows is 1
    """
#     nframes, nchannels, nrows, ncols = image.shape
    nchannels = image.shape[1]
#     nspecies = reference_spectra.shape[0]

    # calculate D matrix using average intensity inverse in each channel to fill diagonal elements
    D = zeros([nchannels, nchannels])
    for ch in range(nchannels):
        D[ch,ch] = image[:,ch][mask].mean()**-1

    # calculate weights from D and reference spectra
    M = reference_spectra
    weights = dot(inv(multi_dot((M, D, M.T))), dot(M, D))  
    
    # return spectrally decomposed image
    return (weights[...,None,None] * image[:,None]).sum(axis=2)
