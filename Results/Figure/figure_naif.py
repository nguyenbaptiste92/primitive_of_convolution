import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

#csvfile = "Inputwidth_Latency_And_Energy.csv"
#save_file = "influence_of_input_width_naive"
#category = "Input width"
#dico_mac = {"Conv" : [2304,0,0], "AddConv" : [2304,0,0], "GroupConv" : [1152,0,0], "DepthwiseConv" : [400,0,0], "ShiftConv" : [256,0,0]}

csvfile = "Inputchannels_Latency_And_Energy.csv"
save_file = "influence_of_input_channels_naive"
category = "Input channels"
dico_mac = {"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,25600,0], "ShiftConv" : [0,16384,0]}

#csvfile = "Kernelsize_Latency_And_Energy.csv"
#save_file = "influence_of_kernel_size_naive"
#category = "Kernel size"
#dico_mac = {"Conv" : [262144,0,0], "AddConv" : [262144,0,0], "GroupConv" : [131072,0,0], "DepthwiseConv" : [16384,0,262144], "ShiftConv" : [0,0,262144]}

#csvfile = "Filters_Latency_And_Energy.csv"
#save_file = "influence_of_filters_naive"
#category = "Filters"
#dico_mac = {"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,16384,147456], "ShiftConv" : [0,16384,0]}

#csvfile = "Groups_Latency_And_Energy.csv"
#save_file = "influence_of_groups_naive"
#category = "Groups

dico_color = {"Conv" : 'g', "AddConv" : 'b', "GroupConv" : 'c', "DepthwiseConv" : 'r', "ShiftConv" : 'm'}
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(15, 6))

ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].set_ylabel('MMACs', fontsize=20)
ax[0].set_xlabel(category, fontsize=20)
ax[0].grid()

ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].set_ylabel('Latency (s)', fontsize=20)
ax[1].set_xlabel(category, fontsize=20)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[1].grid()

ax[2].tick_params(axis='y', labelsize=18)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].set_ylabel('Consumption (mJ)', fontsize=20)
ax[2].set_xlabel(category, fontsize=20)
ax[2].grid()

dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+csvfile,sep=",")

#dsp_dataframe = dataframe[dataframe["DSP"]==1].sort_values(by=[category], ascending=True)
dataframe = dataframe[dataframe["DSP"]==0].sort_values(by=[category], ascending=True)

#dsp_dataframe["Consumption (mJ)"] = dsp_dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ
dataframe["Consumption (mJ)"] = dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ

print(dataframe)
# Compute MMACs
dataframe['coeff'] = dataframe['Layer type'].map(dico_mac)
dataframe['MMACs'] = (dataframe['coeff'].str[0]*dataframe[category]*dataframe[category] + dataframe['coeff'].str[1]*dataframe[category]+ dataframe['coeff'].str[2])/1000000
#dataframe["MMACs"] = (10*10 * 3*3 * 128 * 64) /(dataframe["Groups"]*1000000)

#ax[0].plot(dataframe[category], dataframe["MMACs"],'-o', color = dico_color["GroupConv"])
#ax[1].plot(dataframe[category], dataframe["Latency (s)"],'-o', color = dico_color["GroupConv"])
#ax[2].plot(dataframe[category], dataframe["Consumption (mJ)"],'-o', color = dico_color["GroupConv"])

liste_layer_type = dataframe["Layer type"].sort_values(ascending=True).unique()
# Plot
for j,layer in enumerate(liste_layer_type):

  layer_dataframe = dataframe[dataframe["Layer type"]==layer].sort_values(by=[category], ascending=True)
  
  ax[0].plot(layer_dataframe[category], layer_dataframe["MMACs"],'-o', color = dico_color[layer])
  ax[1].plot(layer_dataframe[category], layer_dataframe["Latency (s)"],'-o', color = dico_color[layer])
  ax[2].plot(layer_dataframe[category], layer_dataframe["Consumption (mJ)"],'-o', color = dico_color[layer])
  
GroupConv = mlines.Line2D([], [], color='c', marker='s', linestyle='None',
                          markersize=12, label='Grouped convolution')

Conv = mlines.Line2D([], [], color='g', marker='s', linestyle='None',
                          markersize=12, label='Convolution')
                          
DepthwiseConv = mlines.Line2D([], [], color='r', marker='s', linestyle='None',
                          markersize=12, label='Depthwise separable convonvolution')
                          
ShiftConv = mlines.Line2D([], [], color='m', marker='s', linestyle='None',
                          markersize=12, label='Shift convolution')
                          
AddConv = mlines.Line2D([], [], color='b', marker='s', linestyle='None',
                          markersize=12, label='Add convolution')

fig.legend(handles=[Conv, AddConv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=5, fontsize= "large",bbox_to_anchor=(0.45, 0.00))

fig.subplots_adjust(wspace=0.35,bottom=0.21)

fig.text(0.22, 0.9, "a)",fontsize = 32)
fig.text(0.50, 0.9, "b)",fontsize = 32)
fig.text(0.78, 0.9, "c)",fontsize = 32)

fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.eps',bbox_inches='tight',pad_inches = 0, format='eps')