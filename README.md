# Fluorescence Covariance Matrix Analysis
This repository contains IPython notebooks (.ipynb files) and supporting Python code (.py files) for completing the analysis described in:  
Foust, D.J., and D.W. Piston. 2021. Measuring G Protein Activation with Spectrally Resolved Fluorescence Fluctuation Spectroscopy. bioRxiv. 2021.11.03.467169.  

To run these notebooks we recommend installation of the Python distribution Anaconda which includes most of the required packages.

IPython Notebooks were tested with the following versions numbers for critical packages:  
python 3.7.6  
anaconda 2020.02  
ipython 7.12.0  
numpy 1.18.1  
scipy 1.4.1  
matplotlib 3.1.3  
ipympl 0.5.6  
tifffile 2020.2.16  

Packages required beyond the standard Anaconda installation: tifffile, ipympl, ipyfilechooser  

Example installations from command line:  
pip install tifffile  
pip install ipympl  
pip install ipyfilechooser  

This analysis assumes multi-dimensional imaging data (time, channel, x, y) stored as .lsm (Zeiss) files. The authors make no promises regarding the maintanence of this code. For questions, concerns, or help getting started, please contact Daniel Foust (dfoust[at]wustl.edu) or David Piston (piston[at]wustl.edu).

## Determining Reference Spectra
1. Combine channels for images of cells expressing a single chromophore: 'combine channels.ipynb'  
2. Draw regions of interest: 'multicolor roi drawing.ipynb'  
3. Calculate average reference spectra: 'calculate reference spectra w masks.ipynb'  

## Unmixing multi-chromophore images
1. Unmix using: 'unmix species.ipynb'  

<img src="fcma%20supporting%20images/spectral unmixing.png" width="500">

## Draw regions of interest for unmixed images
1. Draw regions of interest: 'multicolor roi drawing.ipynb'  

## Calculate detection spectra and covariance matrices
1. Calculate detection spectra and covaraince matrices: 'Calculate Cumulants with Highpass Filter.ipynb'

## Fitting detection spectra and covariance matrices
1. Fit detection spectra and : 'Fit Cumulants.ipynb'

<img src="fcma%20supporting%20images/fitting cumulants.png" width="500">

## Calculate spatial correlation functions (spectrally resolved Raster Image Correlation Spectroscopy)
1. Calculate correlation functions: 'correlation anlaysis w masks frames together.ipynb'

## Fitting spatial autocorrelation functions 
The notebook to complete fitting of autocorrelation functions depends on the number of chromophores unmixed to generate spectrally unmixed imaging data. Select the notebook corresponding to the appropriate number of chromophores.  
Single chromophore: 'correlation fitting one color 2D.ipynb'  
Three chromophores: 'correlation fitting three color 2D.ipynb'
