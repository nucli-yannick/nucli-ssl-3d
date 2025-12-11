from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, TextBox

class Slider:
    def __init__(self, volumes):
        self.volumes = volumes

        self.ind = 0
        self.d = list(self.volumes.values())[0].shape[0]
        self.y_line = 0

        self.fig = plt.figure()
        gs = self.fig.add_gridspec(3, len(volumes.keys()))
        
        self.axs = [[], []]
        for i in range(len(volumes.keys())):
            self.axs[0].append(self.fig.add_subplot(gs[0, i]))
            self.axs[1].append(self.fig.add_subplot(gs[1, i]))
        self.linescan_ax = self.fig.add_subplot(gs[2, :])
        #self.fig, self.axs = plt.subplots(3, len(volumes.keys()))
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.slice_clip = 2.5
        self.region_clip = 2.5
        self.x1 = None

        for i, (m, volume) in enumerate(self.volumes.items()):
            self.axs[0][i].imshow(volume[self.ind], vmax=2.5, cmap='grey')
            self.axs[0][i].set_title(m, loc='left')
        selector = RectangleSelector(self.axs[0][0], self.update_box, useblit=True,
                                    button=[1], interactive=True)

        axbox = self.fig.add_axes([0.05, 0.0, 0.05, 0.05])
        text_box_slice = TextBox(axbox, "full", textalignment="center")
        text_box_slice.on_submit(self.clip_slice)
        text_box_slice.set_val("2.5")


        axbox2 = self.fig.add_axes([0.15, 0.0, 0.05, 0.05])
        text_box = TextBox(axbox2, "reg", textalignment="center")
        text_box.on_submit(self.clip_region)
        text_box.set_val("2.5")

        
        plt.show()

    def clip_slice(self, val):
        self.slice_clip = float(val)
        print(val)
        self.update()

    def clip_region(self, val):
        self.region_clip = float(val)
        print(val)
        self.update_region()

    def on_scroll(self, event):
        ax = event.inaxes
        if (ax in self.axs[0]):
            increment = 1 if event.button == 'up' else -1
            max_index = self.d - 1
            self.ind = np.clip(self.ind + increment, 0, max_index)
            self.update()
        elif self.x1:
            increment = 1 if event.button == 'up' else -1
            max_index = self.y2 - self.y1 - 1
            self.y_line = np.clip(self.y_line + increment, 0, max_index)
            self.update_region()

    def update(self):
        self.linescan_ax.clear()
        for i, (m, volume) in enumerate(self.volumes.items()):
            self.axs[0][i].clear()
            self.axs[0][i].imshow(volume[self.ind], vmax=self.slice_clip, cmap='grey')
            self.axs[0][i].set_title(m, loc='left')


            self.axs[1][i].clear()
            if self.x1:
                sub_slice = volume[self.ind][self.y1:self.y2, self.x1:self.x2]
            else:
                sub_slice = volume[self.ind]
            self.axs[1][i].imshow(sub_slice, cmap='grey', vmax=self.region_clip)
            self.linescan_ax.plot(sub_slice[:, self.y_line], label=m)

        self.linescan_ax.legend()
        plt.draw()

    def update_box(self, eclick, erelease):
        x1, y1, x2, y2 = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
        self.x1, self.x2 = int(min(x1, x2)), int(max(x1, x2))
        self.y1, self.y2 = int(min(y1, y2)), int(max(y1, y2))

        self.update_region()

    def update_region(self):
        self.linescan_ax.clear()
        for i, (m, volume) in enumerate(self.volumes.items()):
            self.axs[1][i].clear()
            if self.x1:
                sub_slice = volume[self.ind][self.y1:self.y2, self.x1:self.x2]
            else:
                sub_slice = volume[self.ind]
            self.axs[1][i].imshow(sub_slice, cmap='grey', vmax=self.region_clip)
            if self.x1:
                self.axs[1][i].plot((0, self.x2-self.x1 - 1), ((self.y_line, self.y_line)), 'red')
            self.linescan_ax.plot(sub_slice[self.y_line, :], label=m)
        self.linescan_ax.legend()
        plt.draw()

import nibabel as nib


volumes = {"Original" : nib.load("/home/vicde/data/reader/azg/fdg/100pc/Omni_FDG_1.nii.gz").get_fdata().astype(np.float16),
                "Back2SS" : np.load("/home/vicde/data/reader_back2SS/azg/fdg/100pc/Omni_FDG_1.npy"), 
                'unet' : np.clip(np.load('/home/vicde/results_unet/reader/azg/fdg/prediction/Omni_FDG_1.npy'), a_min=0, a_max=None)}


Slider(volumes)