# Fluorescence Covariance Matrix Analysis
The depository contains IPython notebooks (.ipynb) and supporting Python code for (.py) completing analysis described in:
Foust, D.J., and D.W. Piston. 2021. Measuring G Protein Activation with Spectrally Resolved Fluorescence Fluctuation Spectroscopy. bioRxiv. 2021.11.03.467169.

IPython Notebooks were tested with the following versions numbers:  
Packages required beyond the standard Anaconda installation: tifffile, ipympl
Example installations from command line:
pip install tifffile
pip install ipympl

## Determining Reference Spectra
1. Combine channels for images of cells expressing a single chromophore: 'combine channels.ipynb'
2. Draw regions of interest: 'multicolor roi drawing.ipynb'
3. Calculate average reference spectra: 'calculate reference spectra w masks.ipynb'

## Unmixing multi-chromophore images
1. Unmix using: 
