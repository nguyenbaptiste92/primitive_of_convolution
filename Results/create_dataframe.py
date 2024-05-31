from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np

from analyse import *

############################################################################################################################################################
####### EXPERIMENT ON NUMBER OF GROUPS
############################################################################################################################################################
"""print("Creation of dataframe for number of groups.")

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Groups"
instruction_types = ("/**/DSP", "/**/No_DSP", "/**/Im2Col")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Groups', 'Latency (s)', 'Consumption (J)'])
for i,folder in enumerate(liste_folder):
    row = []   
    if "No_DSP" in folder:
        row.append(0)
    elif "Im2Col" in folder:
        row.append(1)
    else:
        row.append(2) 
    string = folder.split("Group_")[1].split("/")[0]
    row.append(int(string))
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    
latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Groups_Latency_And_Energy.csv",index=False)"""

############################################################################################################################################################
####### EXPERIMENT ON KERNEL SIZE
############################################################################################################################################################
"""print("Creation of dataframe for kernel size.")

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Kernelsize"
instruction_types = ("/**/DSP", "/**/No_DSP", "/**/Im2Col")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Layer type', 'Kernel size', 'Latency (s)', 'Consumption (J)'])
for i,folder in enumerate(liste_folder):
    row = []   
    if "No_DSP" in folder:
        row.append(0)
    elif "Im2Col" in folder:
        row.append(1)
    else:
        row.append(2) 
    string = folder.split("Kernelsize_")[1].split("/")[1].split("/")[0]
    row.append(string)
    string = folder.split("Kernelsize_")[1].split("/")[0]
    row.append(int(string))
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    
latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Kernelsize_Latency_And_Energy.csv",index=False)"""

############################################################################################################################################################
####### EXPERIMENT ON INPUT WIDTH
############################################################################################################################################################
"""print("Creation of dataframe for input width.")

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Inputwidth"
instruction_types = ("/**/DSP", "/**/No_DSP", "/**/Im2Col")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Layer type', 'Input width', 'Latency (s)', 'Consumption (J)'])
for i,folder in enumerate(liste_folder):
    row = []   
    if "No_DSP" in folder:
        row.append(0)
    elif "Im2Col" in folder:
        row.append(1)
    else:
        row.append(2) 
    string = folder.split("Inputwidth_")[1].split("/")[1].split("/")[0]
    row.append(string)
    string = folder.split("Inputwidth_")[1].split("/")[0]
    row.append(int(string))
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    
latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Inputwidth_Latency_And_Energy.csv",index=False)"""

############################################################################################################################################################
####### EXPERIMENT ON INPUT CHANNEL
############################################################################################################################################################
"""print("Creation of dataframe for input channels.")

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Inputchannels"
instruction_types = ("/**/DSP", "/**/No_DSP", "/**/Im2Col")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Layer type', 'Input channels', 'Latency (s)', 'Consumption (J)'])
for i,folder in enumerate(liste_folder):
    row = []   
    if "No_DSP" in folder:
        row.append(0)
    elif "Im2Col" in folder:
        row.append(1)
    else:
        row.append(2) 
    string = folder.split("Inputchannel_")[1].split("/")[1].split("/")[0]
    row.append(string)
    string = folder.split("Inputchannel_")[1].split("/")[0]
    row.append(int(string))
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    
latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Inputchannels_Latency_And_Energy.csv",index=False)"""

############################################################################################################################################################
####### EXPERIMENT ON FILTERS
############################################################################################################################################################
print("Creation of dataframe for filters.")

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Filters"
instruction_types = ("/**/DSP", "/**/No_DSP", "/**/Im2Col")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Layer type', 'Filters', 'Latency (s)', 'Consumption (J)'])
for i,folder in enumerate(liste_folder):
    row = []   
    if "No_DSP" in folder:
        row.append(0)
    elif "Im2Col" in folder:
        row.append(1)
    else:
        row.append(2) 
    string = folder.split("Outputchannel_")[1].split("/")[1].split("/")[0]
    row.append(string)
    string = folder.split("Outputchannel_")[1].split("/")[0]
    row.append(int(string))
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    
latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Filters_Latency_And_Energy.csv",index=False)

############################################################################################################################################################
####### EXPERIMENT ON FREQUENCY
############################################################################################################################################################
"""print("Creation of 2 dataframes (power, latency and consumption) for frequency.")

intensity_threshold_dico = {"80": 0.01, "40":0.007, "20":0.005, "10":0.0035}
time_threshold_dico = {"80": 0.00, "40":0.0, "20":0.1, "10":0.2} #Sleep mode of 200 ms between inference
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Frequency"
instruction_types = ("/**/DSP", "/**/No_DSP")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Layer type', 'Frequency (MHz)', 'Latency (s)', 'Consumption (J)'])
power_dataframe = pd.DataFrame(columns=['DSP','Layer type','Frequency (MHz)', 'Power (W)'])

for i,folder in enumerate(liste_folder):
    row = []
    row2 = []   
    if "No_DSP" in folder:
        row.append(0)
        row2.append(0)
    else:
        row.append(1)
        row2.append(1)
    string = folder.split("Frequency_")[1].split("/")[1].split("/")[0]
    row.append(string)
    row2.append(string)
    string = folder.split("Frequency_")[1].split("/")[0]
    row.append(int(string))
    row2.append(int(string))
    intensity_threshold = intensity_threshold_dico[string]
    time_threshold = time_threshold_dico[string]
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    power = Get_Inference_AveragePower(dataframe)
    row2.append(power)
    power_dataframe.loc[i] = row2
    
power_series = power_dataframe.groupby(['DSP','Frequency (MHz)']).mean()
power_dataframe = power_series.index.to_frame(name=['DSP','Frequency (MHz)']).reset_index(drop=True)
power_dataframe["Power (W)"] = power_series["Power (W)"].tolist()

latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Frequency_Latency_And_Energy.csv",index=False)
power_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Frequency_Power.csv",index=False)"""

############################################################################################################################################################
####### EXPERIMENT ON OPTIMIZATION LEVEL
############################################################################################################################################################
"""print("Creation of dataframe for optimization level.")

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiment_Optimization"
instruction_types = ("/**/DSP", "/**/No_DSP")

liste_folder = []
for types in instruction_types:
    liste_folder.extend(glob.glob(parent_path+types, recursive = True))

latency_and_consumption_dataframe = pd.DataFrame(columns=['DSP','Optimization level', 'Latency (s)', 'Consumption (J)'])
for i,folder in enumerate(liste_folder):
    row = []   
    if "No_DSP" in folder:
        row.append(0)
    else:
        row.append(1)
    string = folder.split("Optimization_")[1].split("/")[0]
    row.append(string)
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    row.append(consumption)
    latency_and_consumption_dataframe.loc[i] = row
    
latency_and_consumption_dataframe.to_csv(os.getcwd()+os.sep+"Dataframe"+os.sep+"Optimization_Latency_And_Energy.csv",index=False)"""