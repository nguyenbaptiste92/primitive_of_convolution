import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

##############################################################################################################################################################
## DICTIONNARY OF MACs AND MEMORY ACCESS (LOAD+STR instructions)
# The formula comes from the files in the Assembly_Instructions_Count folder
##############################################################################################################################################################

no_dsp_mac_dict = {
    "Conv" : (lambda a,b,c,d,e,f,g : a**2*b*c*d**2),
    "GroupConv" : (lambda a,b,c,d,e,f,g : (a**2*b*c*d**2)/e),
    "ShiftConv" : (lambda a,b,c,d,e,f,g : a**2*b*c),
    "DepthwiseConv" : (lambda a,b,c,d,e,f,g : a**2*b*d**2 + a**2*b*c),
}

no_dsp_memory_dict = {
    "Conv" : (lambda a,b,c,d,e,f,g : (7*a**2*b*c*d**2+12*a**2*c*d**2+7*a**2*c*d+23*a**2*c+8*a*c+4*c+14) + (3*a**2*b*c*d**2+6*a**2*c*d**2+4*a**2*c*d+6*a**2*c+4*a*c+c+18)),
    "GroupConv" : (lambda a,b,c,d,e,f,g : (7*a**2*f*c*d**2+9*a**2*c*d**2+13*a**2*c*d+19*a**2*c+19*a*c+4*c+21)),# + (4*a**2*f*c*d**2+4*a**2*c*d**2+5*a**2*c*d+8*a**2*c+8*a*c+5*c+22)),
    "ShiftConv" : (lambda a,b,c,d,e,f,g : (12*a**2*b*c+5*a**2*c+5*a*c+5*c+5) + (a**2*b*c+a**2*c+a*c+5*c+8)),
    "DepthwiseConv" : (lambda a,b,c,d,e,f,g : (9*a**2*b*d**2+8*a**2*b*d+26*a**2*b+9*a**2+13*a+12) + (3*a**2*b*d**2+2*a**2*b*d+10*a**2*b+4*a**2+6*a+19) + (12*a**2*b*c+5*a**2*c+5*a*c+5*c+5) + (a**2*b*c+a**2*c+a*c+5*c+8)),
}

dsp_mac_dict = {
    "Conv" : (lambda a,b,c,d,e,f,g : (1/2)*a**2*b*c*d**2),
    "GroupConv" : (lambda a,b,c,d,e,f,g : a**2*d**2*e*f*g),
    "ShiftConv" : (lambda a,b,c,d,e,f,g : (1/2)*a**2*b*c),
    "DepthwiseConv" : (lambda a,b,c,d,e,f,g : (1/2)*a**2*b*d**2 + (1/2)*a**2*b + (1/2)*a**2*b*c),
}

dsp_memory_dict = {
    "Conv" : (lambda a,b,c,d,e,f,g : ((7/8)*a**2*b*c*d**2+(1/4)*a**2*b*d**2+(13/2)*a**2*c+7*a**2*d+(19/2)*a**2+4*a+14) + ((3/16)*a**2*b*c*d**2+(1/4)*a**2*b*d**2+(9/4)*a**2*c+a**2*d+8*a**2+5*a+14)),
    "GroupConv" : (lambda a,b,c,d,e,f,g : ((7/8)*a**2*d**2*e*f*g+(1/4)*a**2*d**2*e*f+5*a**2*d**2*e+4*a**2*d*e+(13/2)*a**2*e*g+13*a**2*e+19*a*e+10*e+26)), #+ ((3/16)*a**2*d**2*e*f*g+(1/4)*a**2*d**2*e*f+2*a**2*d**2*e+(9/4)*a**2*e*g+(21/2)*a**2*e+7*a*e+4*e+20)),
    "ShiftConv" : (lambda a,b,c,d,e,f,g : ((7/8)*a**2*b*c+(9/2)*a**2*b+4*a**2*c+14*a**2+2*a+7) + ((5/16)*a**2*b*c+2*a**2*b+(7/4)*a**2*c+(17/2)*a**2+a+6)),
    "DepthwiseConv" : (lambda a,b,c,d,e,f,g : ((17/8)*a**2*b*d**2+4*a**2*d**2+(19/8)*a**2*b+2*a**2*d+20*a**2+9*a+15) + ((7/8)*a**2*b*d**2+a**2*d**2+(7/8)*a**2*b+4*a**2+6*a+23) + ((7/8)*a**2*b*c+(1/4)*a**2*b+4*a**2*c+(23/2)*a**2+a+7) + ((5/16)*a**2*b*c+(1/4)*a**2*b+(7/4)*a**2*c+(21/2)*a**2+a+6)),
}

################################################################################################################################################################
# Plot the ratio between the ratio MAC/Memoryaccess with SIMD and the ratio MAC/Memoryaccess without SIMD
# Green : Convolution
# Red : Depthwise separable convolution
# Cyan : Grouped convolution
# Magenta : Shift convolution
# No add convolution because their is no implementation with SMID instructions
################################################################################################################################################################

list_csvfile = ["Kernelsize_Latency_And_Energy.csv", "Inputwidth_Latency_And_Energy.csv", "Inputchannels_Latency_And_Energy.csv", "Filters_Latency_And_Energy.csv"]
list_category = ["Kernel size", "Input width", "Input channels", "Filters"]

color_dict = {
    "GroupConv" : 'c',
    "Conv" : 'g',
    "DepthwiseConv" : 'r',
    "ShiftConv" : 'm'
}

fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(24, 6))

##############################################################################################################################################################
# Influence of groups for grouped convolutions (only grouped convolutions to plot)
##############################################################################################################################################################

dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Groups_Latency_And_Energy.csv",sep=",")
dataframe["Input width"]=10
dataframe["Input channels"]=128
dataframe["Filters"]=64
dataframe["Kernel size"]=3
dataframe["Channelsbygroup"]=dataframe["Input channels"]/dataframe["Groups"]
dataframe["Filtersbygroup"]=dataframe["Filters"]/dataframe["Groups"]

dataframe["MACfunction"] = np.where(dataframe["DSP"]==0, no_dsp_mac_dict["GroupConv"], dsp_mac_dict["GroupConv"])
dataframe["Memoryfunction"] = np.where(dataframe["DSP"]==0, no_dsp_memory_dict["GroupConv"], dsp_memory_dict["GroupConv"])
MAClist = [row["MACfunction"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Channelsbygroup"],row["Filtersbygroup"]) for i, row in dataframe.iterrows()]
dataframe["MAC"] = MAClist
Memorylist = [row["Memoryfunction"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Channelsbygroup"],row["Filtersbygroup"]) for i, row in dataframe.iterrows()]
dataframe["Memory"] = Memorylist

dataframe["MAC/Memory"] = dataframe["MAC"]/dataframe["Memory"]
print(dataframe)
a = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["MAC/Memory"].apply(list)
ratio_dataframe = a.index.to_frame(index=False)
ratio_dataframe["MAC/Memory"] = a.tolist()
ratio_dataframe[['DSP', 'No_DSP']] = pd.DataFrame(ratio_dataframe["MAC/Memory"].tolist(), index=ratio_dataframe.index)
print(ratio_dataframe)
ratio_dataframe["Ratio"] = ratio_dataframe['DSP']/ratio_dataframe['No_DSP']

###Plot figure
ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].set_ylabel('Ratio', fontsize=20)
ax[0].set_xlabel('Groups', fontsize=20)
ax[0].grid()

ax[0].plot(ratio_dataframe["Groups"], ratio_dataframe["Ratio"],'-o', color = 'c')


##############################################################################################################################################################
# Influence of kernel size, input width, input channels and filters (plot for all layer types)
##############################################################################################################################################################
for i in range(len(list_csvfile)):

    dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+list_csvfile[i],sep=",")
    
    list_layer = list(dataframe.sort_values(by=["Layer type"], ascending=True)[dataframe["DSP"]==1]["Layer type"].unique())
    dataframe = dataframe[dataframe["Layer type"].isin(list_layer)]
    
    ###COMPLETE DATAFRAME
    if list_category[i]!="Input width":
        dataframe["Input width"]=32
    if list_category[i]!="Input channels":
        dataframe["Input channels"]=16
    if list_category[i]!="Filters":
        dataframe["Filters"]=16
    if list_category[i]!="Kernel size":
        dataframe["Kernel size"]=3
    dataframe["Groups"]=2
    dataframe["Channelsbygroup"]=dataframe["Input channels"]/dataframe["Groups"]
    dataframe["Filtersbygroup"]=dataframe["Filters"]/dataframe["Groups"]
    
    ###COMPLETE MAC and MEMORY CALL
    dataframe["MACfunction"] = np.where(dataframe["DSP"]==0, dataframe["Layer type"].map(no_dsp_mac_dict), dataframe["Layer type"].map(dsp_mac_dict))
    dataframe["Memoryfunction"] = np.where(dataframe["DSP"]==0, dataframe["Layer type"].map(no_dsp_memory_dict), dataframe["Layer type"].map(dsp_memory_dict))
    
    MAClist = [row["MACfunction"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Channelsbygroup"],row["Filtersbygroup"]) for i, row in dataframe.iterrows()]
    dataframe["MAC"] = MAClist
    Memorylist = [row["Memoryfunction"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Channelsbygroup"],row["Filtersbygroup"]) for i, row in dataframe.iterrows()]
    dataframe["Memory"] = Memorylist
    
    ###GET ratio between SIMD and No_SIMD MAC/Memory
    dataframe["MAC/Memory"] = dataframe["MAC"]/dataframe["Memory"]
    a = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Layer type",list_category[i]])["MAC/Memory"].apply(list)
    ratio_dataframe = a.index.to_frame(index=False)
    ratio_dataframe["MAC/Memory"] = a.tolist()
    ratio_dataframe[['DSP', 'No_DSP']] = pd.DataFrame(ratio_dataframe["MAC/Memory"].tolist(), index=ratio_dataframe.index)
    ratio_dataframe["Ratio"] = ratio_dataframe['DSP']/ratio_dataframe['No_DSP']
    print(ratio_dataframe)
    
    
    ###Plot figure
    ax[i+1].tick_params(axis='y', labelsize=18)
    ax[i+1].tick_params(axis='x', labelsize=18)
    #ax[i+1].set_ylabel('Ratio', fontsize=20)
    ax[i+1].set_xlabel(list_category[i], fontsize=20)
    ax[i+1].grid()
    
    for j,layer in enumerate(list_layer):
          df = ratio_dataframe[ratio_dataframe["Layer type"]==layer].sort_values(by=[list_category[i]], ascending=True)
          ax[i+1].plot(df[list_category[i]], df["Ratio"],'-o', color = color_dict[layer])
      

GroupConv = mlines.Line2D([], [], color='c', marker='s', linestyle='None',
                          markersize=12, label='Grouped convolution')

Conv = mlines.Line2D([], [], color='g', marker='s', linestyle='None',
                          markersize=12, label='Convolution')
                          
DepthwiseConv = mlines.Line2D([], [], color='r', marker='s', linestyle='None',
                          markersize=12, label='Depthwise separable convonvolution')
                          
ShiftConv = mlines.Line2D([], [], color='m', marker='s', linestyle='None',
                          markersize=12, label='Shift convolution')
                          

fig.legend(handles=[Conv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=4, fontsize= "xx-large",bbox_to_anchor=(0.4, 0.0))
fig.text(0.18, 0.9, "a)",fontsize = 32)
fig.text(0.35, 0.9, "b)",fontsize = 32)
fig.text(0.51, 0.9, "c)",fontsize = 32)
fig.text(0.68, 0.9, "d)",fontsize = 32)
fig.text(0.84, 0.9, "e)",fontsize = 32)
fig.subplots_adjust(wspace=0.35,hspace=0.4,bottom=0.25)

#fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'all_ratiomemoryaccess.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
#fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'all_ratiomemoryaccess.eps',bbox_inches='tight',pad_inches = 0, format='eps')