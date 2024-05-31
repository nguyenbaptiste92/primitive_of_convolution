import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

list_csvfile = ["Inputchannels_Latency_And_Energy.csv", "Inputwidth_Latency_And_Energy.csv", "Kernelsize_Latency_And_Energy.csv", "Filters_Latency_And_Energy.csv"]
list_category = ["Input channels", "Input width", "Kernel size", "Filters"]

################################################################################################################################################################
# List of dictionnary to compute the MACs
# Each element of the list is a dictionary associated with an experiment (first one is for the kernel size)
# A tuple of 3 is associated with each layer and help with the computation of the MACs.
# An exemple is the experiment to see the influence of kernel size:
#       - We fix the number of groups at 2, the input width at 16, the input channels at 16 and the filters at 16.
#       - The MACS for depthwise separable convolutions are H_x^2 * C_x * H_k^2 + H_x^2 * C_x * C_y = (32*32*16) * H_x^2 + (32*32*16*16) = 16384 *  H_x^2 + 262144       
################################################################################################################################################################

liste_dico_macs = [
{"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,25600,0], "ShiftConv" : [0,16384,0]},
{"Conv" : [2304,0,0], "AddConv" : [2304,0,0], "GroupConv" : [1152,0,0], "DepthwiseConv" : [400,0,0], "ShiftConv" : [256,0,0]},
{"Conv" : [262144,0,0], "AddConv" : [262144,0,0], "GroupConv" : [131072,0,0], "DepthwiseConv" : [16384,0,262144], "ShiftConv" : [0,0,262144]},
{"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,16384,147456], "ShiftConv" : [0,16384,0]},
]

################################################################################################################################################################
# Plot MACS,latency and energy consumption(without or with SIMD instructions) against number of groups, kernel size, input width, input channels and filters. Plot the speed up of the use of SIMD instructions
# Blue : Add convolution
# Green : Convolution
# Red : Depthwise separable convolution
# Cyan : Grouped convolution
# Magenta : Shift convolution
################################################################################################################################################################

list_color = ['b','g','r','c','m']
fig, ax = plt.subplots(nrows=3, ncols=5,figsize=(25, 15))

##############################################################################################################################################################
# Influence of groups for grouped convolutions (only grouped convolutions to plot)
##############################################################################################################################################################

ax[0][4].tick_params(axis='y', labelsize=18)
ax[0][4].tick_params(axis='x', labelsize=18)
ax[0][4].set_ylabel('MMACs', fontsize=20)
ax[0][4].set_xlabel('Groups', fontsize=20)
ax[0][4].grid()

ax[1][4].tick_params(axis='y', labelsize=18)
ax[1][4].tick_params(axis='x', labelsize=18)
ax[1][4].set_ylabel('Latency (s)', fontsize=20)
ax[1][4].set_xlabel('Groups', fontsize=20)
ax[1][4].grid()

ax[2][4].tick_params(axis='y', labelsize=18)
ax[2][4].tick_params(axis='x', labelsize=18)
ax[2][4].set_ylabel('Consumption (mJ)', fontsize=20)
ax[2][4].set_xlabel('Groups', fontsize=20)
ax[2][4].grid()

dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Groups_Latency_And_Energy.csv",sep=",")

no_dsp_dataframe = dataframe[dataframe["DSP"]==0].sort_values(by=['Groups'], ascending=True)
no_dsp_dataframe["Consumption (mJ)"] = no_dsp_dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ

# Compute MMACs
no_dsp_dataframe["MMACs"] = (10*10 * 3*3 * 128 * 64) /(no_dsp_dataframe["Groups"]*1000000)  # H_x = 10, C_x = 128, C_y = 64, H_k = 3, we divide by 10e-6 to get MMACs 

# Plot
ax[0][4].plot(no_dsp_dataframe['Groups'], no_dsp_dataframe["MMACs"],'-o', color = 'c')
ax[1][4].plot(no_dsp_dataframe['Groups'], no_dsp_dataframe["Latency (s)"],'-o', color = 'c')
ax[2][4].plot(no_dsp_dataframe['Groups'], no_dsp_dataframe["Consumption (mJ)"],'-o', color = 'c')

##############################################################################################################################################################
# Influence of kernel size, input width, input channels and filters (plot for all layer types)
##############################################################################################################################################################

for i in range(len(list_csvfile)):

    dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+list_csvfile[i],sep=",")
    liste_layer_type = dataframe["Layer type"].sort_values(ascending=True).unique()
    
    dataframe['coeff'] = dataframe['Layer type'].map(liste_dico_macs[i])
    dataframe['MMACs'] = (dataframe['coeff'].str[0]*dataframe[list_category[i]]*dataframe[list_category[i]] + dataframe['coeff'].str[1]*dataframe[list_category[i]] + dataframe['coeff'].str[2])/1000000 # We divide by 10e-6 to get MMACs 
    print(dataframe)
        
    ax[0][i].tick_params(axis='y', labelsize=18)
    ax[0][i].tick_params(axis='x', labelsize=18)
    ax[0][i].set_ylabel('MMACs', fontsize=20)
    ax[0][i].set_xlabel(list_category[i], fontsize=20)
    ax[0][i].grid()
    
    ax[1][i].tick_params(axis='y', labelsize=18)
    ax[1][i].tick_params(axis='x', labelsize=18)
    ax[1][i].set_ylabel('Latency (s)', fontsize=20)
    ax[1][i].set_xlabel(list_category[i], fontsize=20)
    ax[1][i].grid()
    
    ax[2][i].tick_params(axis='y', labelsize=18)
    ax[2][i].tick_params(axis='x', labelsize=18)
    ax[2][i].set_ylabel('Consumption (mJ)', fontsize=20)
    ax[2][i].set_xlabel(list_category[i], fontsize=20)
    ax[2][i].grid()
    
    for j,layer in enumerate(liste_layer_type):
    
      no_dsp_dataframe = dataframe[(dataframe["DSP"]==0) & (dataframe["Layer type"]==layer)].sort_values(by=[list_category[i]], ascending=True)
      no_dsp_dataframe["Consumption (mJ)"] = no_dsp_dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ
      
      ax[0][i].plot(no_dsp_dataframe[list_category[i]], no_dsp_dataframe["MMACs"],'-o', color = list_color[j])
      ax[1][i].plot(no_dsp_dataframe[list_category[i]], no_dsp_dataframe["Latency (s)"],'-o', color = list_color[j])
      ax[2][i].plot(no_dsp_dataframe[list_category[i]], no_dsp_dataframe["Consumption (mJ)"],'-o', color = list_color[j])
      
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

fig.legend(handles=[Conv, AddConv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=5, fontsize= "xx-large",bbox_to_anchor=(0.44, 0.025))

fig.subplots_adjust(wspace=0.35,hspace=0.3,bottom=0.15)

fig.text(0.06, 0.25, "3)",fontsize = 28)
fig.text(0.06, 0.50, "2)",fontsize = 28)
fig.text(0.06, 0.76, "1)",fontsize = 28)
fig.text(0.18, 0.9, "a)",fontsize = 28)
fig.text(0.34, 0.9, "b)",fontsize = 28)
fig.text(0.51, 0.9, "c)",fontsize = 28)
fig.text(0.68, 0.9, "d)",fontsize = 28)
fig.text(0.84, 0.9, "e)",fontsize = 28)

fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'influence_of_layer_parameters.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'influence_of_layer_parameters.eps',bbox_inches='tight',pad_inches = 0, format='eps')