import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

from sklearn.linear_model import LinearRegression

##############################################################################################################################################################
## DICTIONNARY OF INSTRUCTIONS
# The formula comes from the files in the Assembly_Instructions_Count folder
# a = dim_im_out
# b = ch_im_in
# c = ch_im_out
# d = dim_kernel
# g = groups
# p = padding
# h = b/g (channel by group)
# i = c/g (filter by group)
# r = p/d
# s = 1 - r
# t = d-p
# x = a-2p
# y = (a-2p)**2/a**2
# z = 1-y
##############################################################################################################################################################

algo_conversion_dict = {
   0 : "NAIF",
   1 : "IM2COL",
   2 : "SIMD",
}

##############################################################################################################################################################
## GROUPCONVOLUTION
##############################################################################################################################################################

naive_groupconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(2*h+1)+2)+4)+2)+2)),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(4*h+5)+5)+8)+8)+5)+23),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(7*h+8)+11)+19)+18)+7)+22),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*h)+5)+3)+1)+7),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : c*(a*(a*(d*((y*d+z*t)*(16*h+21)+24)+62)+46)+23)+64),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*h*c*d**2),
}

im2col_groupconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(h+4)+2)+1/2*(1/2*i*(2*d**2*h+2)+3)+2)+3)+4*p*(a*(d*(d*(r*(2*h+2)+s*(h+3)+6)+2)+1/2*(1/2*i*(2*d**2*h+2)+1)+2)+3)+5)+4),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*h+1)+1/2*(1/2*i*(4*d**2*h+10)+12)+2)+4)+4*p*(a*(d*(d*(r*h+s*h)+1)+1/2*(1/2*i*(4*d**2*h+10)+12)+2)+4)+10)+20),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(h+4)+4)+1/2*(1/2*i*(9*d**2*h+16)+16)+5)+11)+4*p*(a*(d*(d*(s*h+3)+4)+1/2*(1/2*i*(9*d**2*h+16)+16)+5)+8)+20)+27),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d+1)+d**2*h*i+1)+2)+4*p*(a*(d+d**2*h*i+1)+2)+6)+9),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 0),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(4*h+22)+11)+1/2*(1/2*i*(20*d**2*h+67)+59)+14)+33)+4*p*(a*(d*(d*(r*(4*h+7)+s*(4*h+15)+18)+11)+1/2*(1/2*i*(20*d**2*h+67)+55)+15)+27)+63)+81),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : a**2*h*c*d**2),
}

simd_groupconv_dict = {
    "BRANCH" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(1/2*h+4)+2)+1/2*(1/4*h*i*d**2+i+3)+2)+2)+4*p*(a*(d*(d*(r*(2*h+2)+s*(1/2*h+3)+6)+2)+1/2*(1/4*h*i*d**2+i+1)+2)+3)+1)+4),
    "STR" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(1/4*h+1))+1/2*(5/8*h*i*d**2+7/2*i+16)+2)+7)+4*p*(a*(d*(d*(r*(h+1)+s*(1/4*h+1))+1)+1/2*(5/8*h*i*d**2+7/2*i+16)+2)+4)+5)+21),
    "LOAD" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(1/4*h+4)+4)+1/2*(7/4*h*i*d**2+8*i+19)+5)+19)+4*p*(a*(d*(d*(s*(1/4*h+3)+4)+4)+1/2*(7/4*h*i*d**2+8*i+19)+5)+11)+9)+27),
    "MULT" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d+1)+1)+3)+4*p*(a*(d*(d*s+1)+1)+1)+5)+9),
    "SMLAD": (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*h*c*d**2),
    "TOTAL" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : g*(x*(x*(d*(d*(9/4*h+29)+10)+1/2*(19/4*h*i*d**2+22*i+69)+13)+46)+4*p*(a*(d*(d*(r*(4*h+8)+s*(9/4*h+21)+18)+11)+1/2*(19/4*h*i*d**2+22*i+65)+14)+30)+32)+83),
    "MACS" : (lambda a,b,c,d,g,p,h,i,r,s,t,x,y,z : 1/2*a**2*h*c*d**2),
}

groupconv_dict = {
    "NAIF" : naive_groupconv_dict,
    "IM2COL" : im2col_groupconv_dict,
    "SIMD" : simd_groupconv_dict,
}

##############################################################################################################################################################
## TEST
##############################################################################################################################################################

"""a = 32
b = 16
c = 16
d = 3
p = 1
g = 1
h = b/g
i = c/g
r = p/d
s = 1-r
t = d-p
x = a-2*p
y = (a-2*p)**2/a**2
z = 1-y

for key in shiftconv_dict.keys():
  for instructions in shiftconv_dict[key].keys():
      print(key,instructions)
      print(shiftconv_dict[key][instructions](a,b,c,d,g,p,h,i,r,s,t,x,y,z))"""
      
##############################################################################################################################################################
## PLOT FIGURE FOR GROUP CONVOLUTIONS
##############################################################################################################################################################
      
csvfile = "Groups_Latency_And_Energy.csv"
save_file = "instructions_groups_im2col"
category = "Groups"

dico_color = {"Conv" : 'g', "AddConv" : 'b', "GroupConv" : 'c', "DepthwiseConv" : 'r', "ShiftConv" : 'm'}

dataframe = pd.read_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+csvfile,sep=",")
dataframe["Consumption (mJ)"] = dataframe["Consumption (J)"] * 1000 # We multiply by 10e-3 to get mJ
dataframe["DSP"] = dataframe["DSP"].map(algo_conversion_dict)

dataframe["Input width"]=10
dataframe["Input channels"]=128
dataframe["Filters"]=64
dataframe["Kernel size"]=3
dataframe["Padding"]=1
dataframe["Channelsbygroup"]=dataframe["Input channels"]/dataframe["Groups"]
dataframe["Filtersbygroup"]=dataframe["Filters"]/dataframe["Groups"]
dataframe["r"]=dataframe["Padding"]/dataframe["Kernel size"]
dataframe["s"]=1-dataframe["r"]
dataframe["t"]=dataframe["Kernel size"]-dataframe["Padding"]
dataframe["x"]=dataframe["Input width"]-2*dataframe["Padding"]
dataframe["y"]=dataframe["x"]**2/dataframe["Input width"]**2
dataframe["z"]=1-dataframe["y"]

instruction_tot = [groupconv_dict[row["DSP"]]["TOTAL"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Total_Instructions"] = instruction_tot
load_instruction = [groupconv_dict[row["DSP"]]["LOAD"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Load_Instructions"] = load_instruction
store_instruction = [groupconv_dict[row["DSP"]]["STR"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Store_Instructions"] = store_instruction
mac_instruction = [groupconv_dict[row["DSP"]]["MACS"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
branch_instruction = [groupconv_dict[row["DSP"]]["BRANCH"](row["Input width"],row["Input channels"],row["Filters"],row["Kernel size"],row["Groups"],row["Padding"],row["Channelsbygroup"],row["Filtersbygroup"],row["r"],row["s"],row["t"],row["x"],row["y"],row["z"]) for i, row in dataframe.iterrows()]
dataframe["Branch_Instructions"] = branch_instruction
dataframe["MAC"] = mac_instruction
dataframe["Load_Ratio"] = dataframe["Load_Instructions"]/dataframe["Total_Instructions"]
dataframe["Store_Ratio"] = dataframe["Store_Instructions"]/dataframe["Total_Instructions"]
dataframe["Branch_Ratio"] = dataframe["Branch_Instructions"]/dataframe["Total_Instructions"]
dataframe["MAC_Ratio"] = dataframe["MAC"]/dataframe["Total_Instructions"]
print(dataframe[["DSP", "Groups","Load_Ratio","Store_Ratio","Branch_Ratio","MAC_Ratio"]])

a = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Total_Instructions"].apply(list)
ratio_dataframe = a.index.to_frame(index=False)
ratio_dataframe["Total_Instructions"] = a.tolist()
print(ratio_dataframe)
ratio_dataframe[['SIMD_Instructions','NAIF_Instructions', 'IM2COL_Instructions']] = pd.DataFrame(ratio_dataframe["Total_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Instructions_naif_im2col"] = ratio_dataframe['NAIF_Instructions']/ratio_dataframe['IM2COL_Instructions']

b = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Latency (s)"].apply(list)
ratio_dataframe["Latency (s)"] = b.tolist()
ratio_dataframe[['SIMD_Latency','NAIF_Latency', 'IM2COL_Latency']] = pd.DataFrame(ratio_dataframe["Latency (s)"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Latency_naif_im2col"] = ratio_dataframe['NAIF_Latency']/ratio_dataframe['IM2COL_Latency']

c = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Load_Instructions"].apply(list)
c = pd.Series([x if len(x)==3 else [np.nan]+x for x in c])
ratio_dataframe["Load_Instructions"] = c.tolist()
ratio_dataframe[['SIMD_Load','NAIF_Load', 'IM2COL_Load']] = pd.DataFrame(ratio_dataframe["Load_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Load_naif_im2col"] = ratio_dataframe['NAIF_Load']/ratio_dataframe['IM2COL_Load']

d = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Branch_Instructions"].apply(list)
ratio_dataframe["Branch_Instructions"] = d.tolist()
ratio_dataframe[['SIMD_Branch','NAIF_Branch', 'IM2COL_Branch']] = pd.DataFrame(ratio_dataframe["Branch_Instructions"].tolist(), index=ratio_dataframe.index)
ratio_dataframe["Ratio_Branch_naif_im2col"] = ratio_dataframe['NAIF_Branch']/ratio_dataframe['IM2COL_Branch']

c = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["Branch_Ratio"].apply(list)
ratio_dataframe["Branch_Ratio"] = c.tolist()
ratio_dataframe[['SIMD_Branch_Ratio','NAIF_Branch_Ratio', 'IM2COL_Branch_Ratio']] = pd.DataFrame(ratio_dataframe["Branch_Ratio"].tolist(), index=ratio_dataframe.index)

e = dataframe.sort_values(by=['DSP'], ascending=False).groupby(["Groups"])["MAC_Ratio"].apply(list)
ratio_dataframe["MAC_Ratio"] = e.tolist()
ratio_dataframe[['SIMD_MAC_Ratio','NAIF_MAC_Ratio', 'IM2COL_MAC_Ratio']] = pd.DataFrame(ratio_dataframe["MAC_Ratio"].tolist(), index=ratio_dataframe.index)

print(ratio_dataframe[["Ratio_Latency_naif_im2col","Ratio_Instructions_naif_im2col","Ratio_Branch_naif_im2col",'NAIF_Branch_Ratio', 'IM2COL_Branch_Ratio']])

reg = LinearRegression().fit(ratio_dataframe["Ratio_Instructions_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1))
print(reg.score(ratio_dataframe["Ratio_Instructions_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1)))

reg = LinearRegression().fit(ratio_dataframe["Ratio_Load_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1))
print(reg.score(ratio_dataframe["Ratio_Load_naif_im2col"].values.reshape(-1,1), ratio_dataframe["Ratio_Latency_naif_im2col"].values.reshape(-1,1)))

fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(20, 4))

ax[2].tick_params(axis='y', labelsize=18)
ax[2].tick_params(axis='x', labelsize=18)
ax[2].set_ylabel('Total instructions ratio', fontsize=18)
ax[2].set_ylim([1.0, 4.0])
ax[2].set_xlabel(category, fontsize=20)
ax[2].grid()

ax[0].tick_params(axis='y', labelsize=18)
ax[0].tick_params(axis='x', labelsize=18)
ax[0].set_ylabel('Im2Col MAC instructions (%)', fontsize=18)
ax[0].set_ylim([0.0, 0.5])
ax[0].set_xlabel(category, fontsize=20)
ax[0].grid()

ax[1].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[1].set_ylabel('Im2Col Branch instructions (%)', fontsize=18)
ax[1].set_ylim([0.0, 0.5])
ax[1].set_xlabel(category, fontsize=20)
ax[1].grid()

# Plot 
ax[2].plot(ratio_dataframe[category], ratio_dataframe["Ratio_Instructions_naif_im2col"],'-o', color = 'c')
ax[0].plot(ratio_dataframe[category], ratio_dataframe["IM2COL_MAC_Ratio"],'-o', color = 'c')
ax[1].plot(ratio_dataframe[category], ratio_dataframe['IM2COL_Branch_Ratio'],'-o', color = 'c')

fig.subplots_adjust(wspace=0.35)

fig.text(0.22, 0.9, "a)",fontsize = 32)
fig.text(0.50, 0.9, "b)",fontsize = 32)
fig.text(0.78, 0.9, "c)",fontsize = 32)

fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+save_file+'.eps',bbox_inches='tight',pad_inches = 0, format='eps')

