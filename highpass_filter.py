from numpy import ones, zeros
from scipy.ndimage import convolve
from itertools import product

def highpass_filter(image, window_sz, mask):
    """
    Applies moving average filter along temporal axis.
    image: float, dimensions (time, color, x, y)
    window_sz: window size for moving average in time
    mask: bool, dimensions (time, x, y)
    """
    nframes, nchannels, _, _ = image.shape
    
    # subtract moving average
    window = ones([window_sz,1,1,1])
    image_filt = image - convolve(image, weights=window/window.sum())

    # calculate average value inside mask for each (frame,channel)
    averages = zeros([nframes, nchannels])
    for (frame, channel) in product(range(nframes), range(nchannels)):
        averages[frame,channel] = image[frame,channel][mask[frame]].mean()

    # add back average values
    image_filt += averages[...,None,None]
        
    return image_filt # same dimensions as imput image