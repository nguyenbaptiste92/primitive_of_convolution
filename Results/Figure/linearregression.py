import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression

"""list_csvfile = ["Kernelsize_Latency_And_Energy.csv", "Inputwidth_Latency_And_Energy.csv", "Inputchannels_Latency_And_Energy.csv", "Filters_Latency_And_Energy.csv"]
list_category = ["Kernel size", "Input width", "Input channels", "Filters"]
liste_dico_macs = [
{"Conv" : [262144,0,0], "AddConv" : [262144,0,0], "GroupConv" : [131072,0,0], "DepthwiseConv" : [16384,0,262144], "ShiftConv" : [0,0,262144]},
{"Conv" : [2304,0,0], "AddConv" : [2304,0,0], "GroupConv" : [1152,0,0], "DepthwiseConv" : [400,0,0], "ShiftConv" : [256,0,0]},
{"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,25600,0], "ShiftConv" : [0,16384,0]},
{"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,16384,147456], "ShiftConv" : [0,16384,0]},
]"""

#list_csvfile = ["Inputwidth_Latency_And_Energy.csv"]
#list_category = ["Input width"]
#liste_dico_macs = [{"Conv" : [2304,0,0], "AddConv" : [2304,0,0], "GroupConv" : [1152,0,0], "DepthwiseConv" : [400,0,0], "ShiftConv" : [256,0,0]},]

#list_csvfile = ["Inputchannels_Latency_And_Energy.csv"]
#list_category = ["Input channels"]
#liste_dico_macs = [{"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,25600,0], "ShiftConv" : [0,16384,0]},]

#list_csvfile = ["Kernelsize_Latency_And_Energy.csv"]
#list_category = ["Kernel size"]
#liste_dico_macs = [{"Conv" : [262144,0,0], "AddConv" : [262144,0,0], "GroupConv" : [131072,0,0], "DepthwiseConv" : [16384,0,262144], "ShiftConv" : [0,0,262144]},]

list_csvfile = ["Filters_Latency_And_Energy.csv"]
list_category = ["Filters"]
liste_dico_macs = [{"Conv" : [0,147456,0], "AddConv" : [0,147456,0], "GroupConv" : [0,73728,0], "DepthwiseConv" : [0,16384,147456], "ShiftConv" : [0,16384,0]},]
liste_dataframe = []

##############################################################################################################################################################
# Regroup experiments_dataframe with optimization level -Os and frequency 84 MHz.
##############################################################################################################################################################


# Influence of groups for grouped convolutions
#dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Groups_Latency_And_Energy.csv",sep=",")
#dataframe["MACs"] = (10*10 * 3*3 * 128 * 64) /dataframe["Groups"]  # H_x = 10, C_x = 128, C_y = 64, H_k = 3
#liste_dataframe.append(dataframe[["DSP","Latency (s)","Consumption (J)","MACs"]])

# Influence of kernel size, input width, input channels and filters

for i in range(len(list_csvfile)):

    dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+list_csvfile[i],sep=",")
    
    dataframe['coeff'] = dataframe['Layer type'].map(liste_dico_macs[i])
    dataframe['MACs'] = (dataframe['coeff'].str[0]*dataframe[list_category[i]]*dataframe[list_category[i]] + dataframe['coeff'].str[1]*dataframe[list_category[i]] + dataframe['coeff'].str[2])
    liste_dataframe.append(dataframe[['Layer type',"DSP","Latency (s)","Consumption (J)","MACs"]])

big_dataframe = pd.concat(liste_dataframe,ignore_index=True,sort=False)

dsp_dataframe = big_dataframe[(big_dataframe["DSP"]==1)]
no_dsp_dataframe = big_dataframe[(big_dataframe["DSP"]==0)]

print(no_dsp_dataframe)

##################################################################################################################################################################
# No SIMD instructions
##################################################################################################################################################################
print("No SIMD : MACs to Consumption")
reg1 = LinearRegression().fit(np.array(no_dsp_dataframe['MACs']).reshape(-1, 1), no_dsp_dataframe['Consumption (J)'])
print(reg1.score(np.array(no_dsp_dataframe['MACs']).reshape(-1, 1), no_dsp_dataframe['Consumption (J)']))
print(reg1.coef_)
print(reg1.intercept_)

print("No SIMD : MACs to Latency")
reg2 = LinearRegression().fit(np.array(no_dsp_dataframe['MACs']).reshape(-1, 1), no_dsp_dataframe['Latency (s)'])
print(reg2.score(np.array(no_dsp_dataframe['MACs']).reshape(-1, 1), no_dsp_dataframe['Latency (s)']))
print(reg2.coef_)
print(reg2.intercept_)

##################################################################################################################################################################
# SIMD instructions
##################################################################################################################################################################
print("SIMD : MACs to Consumption")
reg3 = LinearRegression().fit(np.array(dsp_dataframe['MACs']).reshape(-1, 1), dsp_dataframe['Consumption (J)'])
print(reg3.score(np.array(dsp_dataframe['MACs']).reshape(-1, 1), dsp_dataframe['Consumption (J)']))
print(reg3.coef_)
print(reg3.intercept_)

print("SIMD : MACs to Latency")
reg4 = LinearRegression().fit(np.array(dsp_dataframe['MACs']).reshape(-1, 1), dsp_dataframe['Latency (s)'])
print(reg4.score(np.array(dsp_dataframe['MACs']).reshape(-1, 1), dsp_dataframe['Latency (s)']))
print(reg4.coef_)
print(reg4.intercept_)
