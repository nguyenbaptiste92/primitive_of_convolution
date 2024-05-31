import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

#csvfile = "Inputwidth_Latency_And_Energy.csv"
#save_file = "influence_of_input_width_im2col"
#category = "Input width"

#csvfile = "Inputchannels_Latency_And_Energy.csv"
#save_file = "influence_of_input_channels_im2col"
#category = "Input channels"

#csvfile = "Kernelsize_Latency_And_Energy.csv"
#save_file = "influence_of_kernel_size_im2col"
#category = "Kernel size"

#csvfile = "Filters_Latency_And_Energy.csv"
#save_file = "influence_of_filters_im2col"
#category = "Filters"

csvfile = "Groups_Latency_And_Energy.csv"
save_file = "influence_of_groups_im2col"
category = "Groups"

dico_color = {"Conv" : 'g', "AddConv" : 'b', "GroupConv" : 'c', "DepthwiseConv" : 'r', "ShiftConv" : 'm'}
fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(20, 4),sharex=True)

ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].set_ylabel('Latency (s)', fontsize=20)
ax[0].set_xlabel(category, fontsize=20)
ax[0].grid()

ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].set_ylabel('Latency gain', fontsize=20)
ax[1].set_xlabel(category, fontsize=20)
ax[1].grid()

ax[2].tick_params(axis='y', labelsize=18)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].set_ylabel('Consumption (mJ)', fontsize=20)
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[2].set_xlabel(category, fontsize=20)
ax[2].grid()

ax[3].tick_params(axis='y', labelsize=18)
ax[3].tick_params(axis='x', labelsize=18)
ax[3].set_ylabel('Consumption gain', fontsize=20)
ax[3].set_xlabel(category, fontsize=20)
ax[3].grid()

dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+csvfile,sep=",")
dataframe["Consumption (mJ)"] = dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ

im2col_dataframe = dataframe[dataframe["DSP"]==1].sort_values(by=[category], ascending=True)
naive_dataframe = dataframe[dataframe["DSP"]==0].sort_values(by=[category], ascending=True)

a = im2col_dataframe[[category,"Latency (s)"]].rename({category: category, "Latency (s)": "im2col_Latency (s)"}, axis=1)
b = naive_dataframe[[category,"Latency (s)"]].rename({category: category, "Latency (s)": "naive_Latency (s)"}, axis=1)
b = b.set_index(category).join(a.set_index(category))

b["Latency gain"] = b["naive_Latency (s)"]/b["im2col_Latency (s)"]
b.dropna(subset=["Latency gain"])

c = im2col_dataframe[[category,"Consumption (J)"]].rename({category: category, "Consumption (J)": "im2col_Consumption (J)"}, axis=1)
d = naive_dataframe[[category,"Consumption (J)"]].rename({category: category, "Consumption (J)": "naive_Consumption (J)"}, axis=1)
d = d.set_index(category).join(c.set_index(category))

d["Consumption_gain"] = d["naive_Consumption (J)"]/d["im2col_Consumption (J)"]
d.dropna(subset=["Consumption_gain"])


ax[0].plot(im2col_dataframe[category], im2col_dataframe["Latency (s)"],'-o', color = dico_color["GroupConv"])
ax[2].plot(im2col_dataframe[category], im2col_dataframe["Consumption (mJ)"],'-o', color = dico_color["GroupConv"])
if len(b)!=0:
  ax[1].plot(naive_dataframe[category], b["Latency gain"],'-o', color = dico_color["GroupConv"])
if len(d)!=0:
  ax[3].plot(naive_dataframe[category], d["Consumption_gain"],'-o', color = dico_color["GroupConv"])

"""liste_layer_type = dataframe["Layer type"].sort_values(ascending=True).unique()
# Plot
for j,layer in enumerate(liste_layer_type):

  layer_im2col_dataframe = im2col_dataframe[im2col_dataframe["Layer type"]==layer].sort_values(by=[category], ascending=True)
  layer_naive_dataframe = naive_dataframe[naive_dataframe["Layer type"]==layer].sort_values(by=[category], ascending=True)
  
  a = layer_im2col_dataframe[[category,"Latency (s)"]].rename({category: category, "Latency (s)": "im2col_Latency (s)"}, axis=1)
  b = layer_naive_dataframe[[category,"Latency (s)"]].rename({category: category, "Latency (s)": "naive_Latency (s)"}, axis=1)
  b = b.set_index(category).join(a.set_index(category))
  
  b["Latency gain"] = b["naive_Latency (s)"]/b["im2col_Latency (s)"]
  b.dropna(subset=["Latency gain"])
  
  c = layer_im2col_dataframe[[category,"Consumption (J)"]].rename({category: category, "Consumption (J)": "im2col_Consumption (J)"}, axis=1)
  d = layer_naive_dataframe[[category,"Consumption (J)"]].rename({category: category, "Consumption (J)": "naive_Consumption (J)"}, axis=1)
  d = d.set_index(category).join(c.set_index(category))
  
  d["Consumption_gain"] = d["naive_Consumption (J)"]/d["im2col_Consumption (J)"]
  d.dropna(subset=["Consumption_gain"])
  
  ax[0].plot(layer_im2col_dataframe[category], layer_im2col_dataframe["Latency (s)"],'-o', color = dico_color[layer])
  ax[2].plot(layer_im2col_dataframe[category], layer_im2col_dataframe["Consumption (mJ)"],'-o', color = dico_color[layer])
  if len(b)!=0:
    ax[1].plot(layer_naive_dataframe[category], b["Latency gain"],'-o', color = dico_color[layer])
  if len(d)!=0:
    ax[3].plot(layer_naive_dataframe[category], d["Consumption_gain"],'-o', color = dico_color[layer])
  
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

fig.legend(handles=[Conv, AddConv, DepthwiseConv, GroupConv, ShiftConv],loc="lower center",ncol=5, fontsize= "large",bbox_to_anchor=(0.42, 0.00))"""

fig.subplots_adjust(wspace=0.35,bottom=0.3)

fig.text(0.20, 0.9, "a)",fontsize = 32)
fig.text(0.40, 0.9, "b)",fontsize = 32)
fig.text(0.60, 0.9, "c)",fontsize = 32)
fig.text(0.80, 0.9, "d)",fontsize = 32)


fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.eps',bbox_inches='tight',pad_inches = 0, format='eps')