from os.path import split, join, exists
from os import listdir, makedirs
import numpy as np
from tempfile import mkdtemp
from scipy.ndimage import gaussian_filter
from ipywidgets import IntSlider, FloatRangeSlider, HBox, Button
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from skimage.draw import polygon
from glob import glob
from pathlib import Path
from skimage.filters import threshold_otsu

class RoiDrawer:
    def __init__(self, filename, maskfld='masks'):
        self.filename  = filename
        self.directory = split(filename)[0]
        self.filelist = glob(join(self.directory, '*.npy'))
        self.maskfld   = maskfld
        
#         self.image = np.load(self.filename, mmap_mode='r')
        self.image = np.load(self.filename)
        self.numframes = self.image.shape[0]
        self.numchannels = self.image.shape[1]
        self.current_frame = 0
        
        self.blur_radius = 4
#         self.image_blur = np.memmap(join(mkdtemp(),'newfile.dat'), 
#                                     shape=self.image.shape, mode='w+', dtype='float64')
        self.image_blur = gaussian_filter(self.image.astype('float'), sigma=(0,0,self.blur_radius,self.blur_radius))
        
        self.init_ths = np.array([threshold_otsu(self.image_blur[:,ch]) for ch in range(self.numchannels)]) 
        self.composite_mask = np.ones([self.image.shape[0], self.image.shape[2], self.image.shape[3]], 
                                      dtype='bool')
#         self.composite_mask_nan = np.ones([self.image.shape[0], self.image.shape[2], self.image.shape[3]]) * np.nan
        self.composite_mask_nan = np.empty([self.image.shape[0], self.image.shape[2], self.image.shape[3]])
        self.composite_mask_nan.fill(np.nan)
        self.blur_masks = np.ones(self.image.shape, dtype='bool')
        self.poly_masks = np.ones([self.numchannels, self.image.shape[2], self.image.shape[3]], dtype='bool')
        
        # define interactives
        self.frame_slider = IntSlider(value=0, min=0, max=self.numframes-1, 
                                      description='Frame:', 
                                      on_trait_change=self.change_frame)
        self.frame_slider.observe(self.change_frame, names='value')
        
        # initialize raw contrast sliders
        self.rawsliders = [FloatRangeSlider(value=[np.percentile(self.image[:,ch], 5), 
                               np.percentile(self.image[:,ch], 95)],
                                min=self.image[:,ch].min(),
                                max=self.image[:,ch].max(),          
                                description='Raw Cont:') for ch in range(self.numchannels)]
        display(HBox(self.rawsliders))
        
        self.blursliders = [FloatRangeSlider(value=[np.percentile(self.image_blur[:,ch], 5), 
                               np.percentile(self.image_blur[:,ch], 95)],
                                min=self.image_blur[:,ch].min(),
                                max=self.image_blur[:,ch].max(),
                                description='Bl Cont:') for ch in range(self.numchannels)]
        display(HBox(self.blursliders))
        
        self.threshsliders = [FloatRangeSlider(value=[self.image_blur[:,ch].min(), 
                               self.image_blur[:,ch].max()],
                                min=self.image_blur[:,ch].min(),
                                max=self.image_blur[:,ch].max(),
                                description='Range:') for ch in range(self.numchannels)]
        display(HBox(self.threshsliders))
        display(self.frame_slider)
    
        # initialize figure
        self.fig, self.ax = plt.subplots(2, self.numchannels, 
                                         figsize=(2*self.numchannels, 4.2), facecolor='xkcd:white')
        if self.ax.ndim == 1:
            self.ax = self.ax.reshape([2,1])
        [axi.set_axis_off() for axi in self.ax.ravel()]
        self.title = self.fig.suptitle(split(self.filename)[1], y=1.00)
        
        self.curraxis = 0
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        self.ims_handles = [[[] for i in range(self.numchannels)] for j in range(2)]
        self.mask_handles = [[[] for i in range(self.numchannels)] for j in range(2)]
        
        for ch in range(self.numchannels):
            self.ims_handles[0][ch] = self.ax[0,ch].imshow(self.image[0,ch], interpolation='none',
                                                          vmin=np.percentile(self.image[:,ch], 5),
                                                          vmax=np.percentile(self.image[:,ch], 95))
            self.ims_handles[1][ch] = self.ax[1,ch].imshow(self.image_blur[0,ch], interpolation='none',
                                                          vmin=np.percentile(self.image_blur[:,ch], 5),
                                                          vmax=np.percentile(self.image_blur[:,ch], 95))
            self.mask_handles[0][ch] = self.ax[0,ch].imshow(self.composite_mask_nan[0], 
                                                            interpolation='none', alpha=0.4, cmap='binary')
            self.mask_handles[1][ch] = self.ax[1,ch].imshow(self.composite_mask_nan[0], 
                                                            interpolation='none', alpha=0.4, cmap='binary')
        plt.tight_layout()
        
        self.channel_idx = np.arange(self.numchannels)
        
        for channel, (rawslider, blurslider, threshslider) in enumerate(zip(self.rawsliders, self.blursliders, self.threshsliders)):
            rawslider.observe(lambda change, channel=channel: self.change_clims(change, channel, 0), names='value')
            blurslider.observe(lambda change, channel=channel: self.change_clims(change, channel, 1), names='value')
            threshslider.observe(lambda change, channel=channel: self.change_ths(change, channel), names='value')
        self.axdict = {}
        for i, ax in enumerate(self.ax[0,:]):
            self.axdict[ax] = i
        
        self.polygons = [PolygonSelector(self.ax[0,ch], onselect=self.on_poly_select, useblit=True,
                                          lineprops={'color': 'xkcd:neon green', 'linewidth': 1.5},
                                          markerprops={'markerfacecolor': 'xkcd:neon green', 
                                                       'markersize': 5,
                                                       'alpha': 1}) for ch in range(self.numchannels)]
        
        self.reset_button = Button(description='Reset')
        self.reset_button.on_click(self.on_reset)
        self.save_button = Button(description='Save')
        self.save_button.on_click(self.on_save)
        self.next_button = Button(description='Next')
        self.next_button.on_click(self.on_next)
        self.savenext_button = Button(description='Save&Next')
        self.savenext_button.on_click(self.on_save)
        self.savenext_button.on_click(self.on_next)
        self.prev_button = Button(description='Previous')
        self.prev_button.on_click(self.on_prev)
        display(HBox([self.reset_button, 
                      self.prev_button,
                      self.save_button,
                      self.savenext_button,
                      self.next_button]))

    def update_mask(self):
        self.composite_mask[:] = self.poly_masks.all(axis=0)*np.ones([self.numframes,1,1], dtype='bool') & self.blur_masks.all(axis=1)
        self.composite_mask_nan[~self.composite_mask] = 1
        self.composite_mask_nan[self.composite_mask] = np.nan
        for ch in range(self.numchannels):
            self.mask_handles[0][ch].set_data(self.composite_mask_nan[self.current_frame])
            self.mask_handles[1][ch].set_data(self.composite_mask_nan[self.current_frame])        
        
    def on_poly_select(self, pos):
        verts = np.asarray(pos)
        rr, cc = polygon(verts[:,1], verts[:,0])
        idx = self.axdict[self.curraxis]
        self.poly_masks[idx] = False
        self.poly_masks[idx][rr,cc] = True
        self.update_mask()
        
    def onclick(self, event):
        self.curraxis = event.inaxes

    def change_frame(self, frame):
        self.current_frame = frame['new']
        for ch in range(self.numchannels):
            self.ims_handles[0][ch].set_data(self.image[self.current_frame, ch])
            self.ims_handles[1][ch].set_data(self.image_blur[self.current_frame, ch])
            self.mask_handles[0][ch].set_data(self.composite_mask_nan[self.current_frame])
            self.mask_handles[1][ch].set_data(self.composite_mask_nan[self.current_frame])
            
    def change_clims(self, clims, channel, row):
        self.ims_handles[row][channel].set_clim(clims['new'])
                
    def change_ths(self, lims, channel):
        low = lims['new'][0]; high = lims['new'][1]
        self.blur_masks[:,channel] = (self.image_blur[:,channel] >= low) & (self.image_blur[:,channel] <= high)
        self.update_mask()
        
    def reset_mask(self):
        self.composite_mask[:] = np.ones([self.image.shape[0], self.image.shape[2], self.image.shape[3]], 
                                      dtype='bool')
        self.composite_mask_nan[:] = np.ones([self.image.shape[0], self.image.shape[2], self.image.shape[3]], 
                                      dtype='bool') * np.nan
        self.blur_masks[:] = np.ones(self.image.shape, dtype='bool')
        self.poly_masks[:] = np.ones([self.numchannels, self.image.shape[2], self.image.shape[3]], dtype='bool')
        
        [pg.set_visible(False) for pg in self.polygons]
        [pg.disconnect_events() for pg in self.polygons]
        
        self.polygons = [PolygonSelector(self.ax[0,ch], onselect=self.on_poly_select, useblit=True,
                                          lineprops={'color': 'xkcd:neon green', 'linewidth': 1.5},
                                          markerprops={'markerfacecolor': 'xkcd:neon green', 
                                                       'markersize': 5,
                                                      'alpha': 1}) for ch in range(self.numchannels)]
        self.update_mask()
        
    def on_reset(self, b):
        self.reset_mask()
        
    def on_next(self, b):
        nextindex = self.filelist.index(self.filename) + 1
        if nextindex == 0 or nextindex == len(self.filelist):
            print('End of Directory')
        else:
            self.filename = join(self.directory, self.filelist[nextindex])
            self.image = np.load(self.filename, mmap_mode='r')
            self.image_blur[:] = gaussian_filter(self.image.astype('float'), 
                                                 sigma=(0,0, self.blur_radius, self.blur_radius))
            self.reset_image()
            self.reset_sliders()
            self.reset_mask()
            self.title.set_text(split(self.filename)[1])
            
    def on_prev(self, b):
        previndex = self.filelist.index(self.filename) - 1
        if previndex < 0:
            print('Start of Directory')
        else:
            self.filename = join(self.directory, self.filelist[previndex])
            self.image = np.load(self.filename, mmap_mode='r')
            self.image_blur[:] = gaussian_filter(self.image.astype('float'), 
                                                 sigma=(0,0, self.blur_radius, self.blur_radius))
            self.reset_image()
            self.reset_sliders()
            self.reset_mask()
            self.title.set_text(split(self.filename)[1])
            
    def on_save(self, b):
        if not exists(join(self.directory, self.maskfld)):
            makedirs(join(self.directory, self.maskfld))
        numrois = len(glob(join(self.directory, self.maskfld, Path(self.filename).stem+' ROI??.npy')))
        np.save(join(self.directory, self.maskfld, Path(self.filename).stem+' ROI'+str(numrois+1).zfill(2)+'.npy'),
               self.composite_mask)
            
    def reset_image(self):
        self.ims_handles = [[[] for i in range(self.numchannels)] for j in range(2)]
        self.mask_handles = [[[] for i in range(self.numchannels)] for j in range(2)]
        for ch in range(self.numchannels):
            self.ims_handles[0][ch] = self.ax[0,ch].imshow(self.image[0,ch], interpolation='none',
                                                          vmin=np.percentile(self.image[:,ch], 5),
                                                          vmax=np.percentile(self.image[:,ch], 95))
            self.ims_handles[1][ch] = self.ax[1,ch].imshow(self.image_blur[0,ch], interpolation='none',
                                                          vmin=np.percentile(self.image_blur[:,ch], 5),
                                                          vmax=np.percentile(self.image_blur[:,ch], 95))
            self.mask_handles[0][ch] = self.ax[0,ch].imshow(self.composite_mask_nan[0], 
                                                            interpolation='none', alpha=0.4, cmap='binary')
            self.mask_handles[1][ch] = self.ax[1,ch].imshow(self.composite_mask_nan[0], 
                                                            interpolation='none', alpha=0.4, cmap='binary')
        self.reset_sliders()
            
    def reset_sliders(self):
        for channel in range(self.numchannels):
            newrawmin = self.image[:,channel].min()
            if newrawmin > self.rawsliders[channel].max:
                self.rawsliders[channel].max = self.image[:,channel].max()
                self.rawsliders[channel].min = self.image[:,channel].min()
            else:
                self.rawsliders[channel].min = self.image[:,channel].min()
                self.rawsliders[channel].max = self.image[:,channel].max()
            self.rawsliders[channel].value = [np.percentile(self.image[:,channel], 5), 
                                              np.percentile(self.image[:,channel], 95)]
                
            newblurmin = self.image_blur[:,channel].min()
            if newblurmin > self.blursliders[channel].max:
                self.blursliders[channel].max = self.image_blur[:,channel].max()
                self.blursliders[channel].min = self.image_blur[:,channel].min()
                self.threshsliders[channel].max = self.image_blur[:,channel].max()
                self.threshsliders[channel].min = self.image_blur[:,channel].min()
            else:
                self.blursliders[channel].min = self.image_blur[:,channel].min()
                self.blursliders[channel].max = self.image_blur[:,channel].max() 
                self.threshsliders[channel].min = self.image_blur[:,channel].min()
                self.threshsliders[channel].max = self.image_blur[:,channel].max()
            self.blursliders[channel].value = [np.percentile(self.image_blur[:,channel], 5), 
                                              np.percentile(self.image_blur[:,channel], 95)]
            self.threshsliders[channel].value = [self.image_blur[:,channel].min(), self.image_blur[:,channel].max()]
            
            self.frame_slider.value = 0; self.current_frame = 0