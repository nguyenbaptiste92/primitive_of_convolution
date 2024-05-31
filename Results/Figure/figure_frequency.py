import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Open dataframe
csv_file = os.getcwd()+os.sep+"Dataframe"+os.sep+"Frequency_Latency_And_Energy.csv"
dataframe = pd.read_csv(csv_file,sep=",")

liste_layer_type = dataframe["Layer type"].sort_values(ascending=True).unique()
dataframe['Consumption (mJ)'] = dataframe['Consumption (J)'] * 1000 #* 1000 to convert J to mJ

################################################################################################################################################################
# Plot latency and energy consumption against frequecy (without or with SIMD instructions)
# Blue : Add convolution
# Green : Convolution
# Red : Depthwise separable convolution
# Cyan : Grouped convolution
# Magenta : Shift convolution
################################################################################################################################################################
list_color = ['b','g','r','c','m']

fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(16, 4))
ax[0].tick_params(axis='y', labelsize=12)
ax[0].set_ylabel('Latency (s)', fontsize=16)
ax[0].set_xlabel('Frequency (MHz)', fontsize=16)
ax[0].grid()

ax[1].tick_params(axis='y', labelsize=12)
ax[1].set_ylabel('Consumption (mJ)', fontsize=16)
ax[1].set_xlabel('Frequency (MHz)', fontsize=16)
ax[1].grid()

ax[2].tick_params(axis='y', labelsize=12)
ax[2].set_ylabel('Latency (s)', fontsize=16)
ax[2].set_xlabel('Frequency (MHz)', fontsize=16)
ax[2].grid()

ax[3].tick_params(axis='y', labelsize=12)
ax[3].set_ylabel('Consumption (mJ)', fontsize=16)
ax[3].set_xlabel('Frequency (MHz)', fontsize=16)
ax[3].grid()

for j,layer in enumerate(liste_layer_type):
    
      dsp_dataframe = dataframe[(dataframe["DSP"]==1) & (dataframe["Layer type"]==layer)].sort_values(by=['Frequency (MHz)'], ascending=True)
      no_dsp_dataframe = dataframe[(dataframe["DSP"]==0) & (dataframe["Layer type"]==layer)].sort_values(by=['Frequency (MHz)'], ascending=True)
      
      ax[0].plot(no_dsp_dataframe['Frequency (MHz)'], no_dsp_dataframe['Latency (s)'],'-o', color = list_color[j])
      ax[1].plot(no_dsp_dataframe['Frequency (MHz)'], no_dsp_dataframe['Consumption (mJ)'],'-o', color = list_color[j])
      ax[2].plot(dsp_dataframe['Frequency (MHz)'], dsp_dataframe['Latency (s)'],'-o', color = list_color[j])
      ax[3].plot(dsp_dataframe['Frequency (MHz)'], dsp_dataframe['Consumption (mJ)'],'-o', color = list_color[j])

GroupConv = mlines.Line2D([], [], color='c', marker='s', linestyle='None',
                          markersize=12, label='Group convolution')

Conv = mlines.Line2D([], [], color='g', marker='s', linestyle='None',
                          markersize=12, label='Convolution')
                          
DepthwiseConv = mlines.Line2D([], [], color='r', marker='s', linestyle='None',
                          markersize=12, label='Depthwise separable convonvolution')
                          
ShiftConv = mlines.Line2D([], [], color='m', marker='s', linestyle='None',
                          markersize=12, label='Shift convolution')
                          
AddConv = mlines.Line2D([], [], color='b', marker='s', linestyle='None',
                          markersize=12, label='Add convolution')

fig.legend(handles=[Conv, AddConv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=5, fontsize= "large",bbox_to_anchor=(0.44, 0.0))

fig.subplots_adjust(wspace=0.40,bottom = 0.25)

fig.text(0.19, 0.9, "a)",fontsize = 28)
fig.text(0.40, 0.9, "b)",fontsize = 28)
fig.text(0.61, 0.9, "c)",fontsize = 28)
fig.text(0.82, 0.9, "d)",fontsize = 28)

fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'influence_of_frequency.jpeg', bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'influence_of_frequency.eps', bbox_inches='tight',pad_inches = 0, format='eps')