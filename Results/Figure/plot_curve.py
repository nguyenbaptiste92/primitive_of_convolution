from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

from analyse import *

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Kernelsize"
instruction_types = ("/**/DSP", "/**/No_DSP", "/**/Im2Col")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))


liste_dataframe = []
for i,folder in enumerate(liste_folder):
    if "Kernelsize_3" in folder and "No_DSP" in folder:
        print(folder)
        liste_dataframe.append(Unzip_Experiment_To_Dataframe(folder))       

fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))

ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].set_ylabel('Intensity', fontsize=20)
ax[0].set_xlabel('Time (s)', fontsize=20)
ax[0].set_ylim([0.005, 0.03])
ax[0].grid()

ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].set_ylabel('Intensity', fontsize=20)
ax[1].set_xlabel('Time (s)', fontsize=20)
ax[1].set_ylim([0.005, 0.03])
ax[1].grid()

ax[2].tick_params(axis='y', labelsize=18)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].set_ylabel('Intensity', fontsize=20)
ax[2].set_xlabel('Time (s)', fontsize=20)
ax[2].set_ylim([0.005, 0.03])
ax[2].grid()

dataframe = liste_dataframe[2]
dataframe = dataframe[(dataframe["Time (s)"]>1.0) & (dataframe["Time (s)"]<3.0)] 
ax[0].plot(dataframe["Time (s)"], dataframe["Intensity (A)"])

dataframe = liste_dataframe[3]
dataframe = dataframe[(dataframe["Time (s)"]>1.0) & (dataframe["Time (s)"]<3.0)] 
ax[1].plot(dataframe["Time (s)"], dataframe["Intensity (A)"])

dataframe = liste_dataframe[0]
dataframe = dataframe[(dataframe["Time (s)"]>1.0) & (dataframe["Time (s)"]<3.0)] 
ax[2].plot(dataframe["Time (s)"], dataframe["Intensity (A)"])

fig.subplots_adjust(wspace=0.4)

fig.text(0.22, 0.9, "a)",fontsize = 32)
fig.text(0.50, 0.9, "b)",fontsize = 32)
fig.text(0.78, 0.9, "c)",fontsize = 32)

fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'naive_intensity.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'naive_intensity.eps',bbox_inches='tight',pad_inches = 0, format='eps')