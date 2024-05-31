from pathlib import Path
import os
import glob
from pyunpack import Archive
import pandas as pd
import numpy as np

def Unzip_Experiment_To_Dataframe(path):

    # Unzip stpm file
    path = path+os.sep
    for file_name in glob.glob(path+'*.stpm'):
      if not os.path.exists(path+'unzippedFiles'):
        os.mkdir(path+'unzippedFiles')
      Archive(file_name).extractall(path+'unzippedFiles')
 
    # Read all csv files 
    frames = []  
    path = path+'unzippedFiles'+os.sep
    for file_name in glob.glob(path+'*rawfile*.csv'):
      dataframe = pd.read_csv(file_name, sep=";")
      dataframe.columns =['Time', 'Intensity']
      frames.append(dataframe)
      
    # Process all the dataframes into one
    dataframe = pd.concat(frames)
    dataframe = dataframe.sort_values(by=['Time'], ascending=True)
    dataframe = dataframe.reset_index(drop=True)
    
    # Put values in SI 
    dataframe["Time (s)"] = dataframe["Time"]/1000
    dataframe["Intensity (A)"] = dataframe["Intensity"]*1e-6
    dataframe["Power (W)"] = dataframe["Intensity (A)"]*3.3
    
    return dataframe[["Time (s)", "Intensity (A)", "Power (W)"]]
    
def Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold):

    # Add inference column
    dataframe["Is_Inference"] = np.where(dataframe["Intensity (A)"] > intensity_threshold, 1, 0)
    dataframe["Inference_begin"] = np.where((dataframe["Is_Inference"]==1) & (dataframe["Is_Inference"].shift(1)==0), 1, 0)
    dataframe["Inference_number"] = dataframe["Inference_begin"].cumsum()
    dataframe = dataframe.loc[(dataframe["Is_Inference"]==1)]
    
    #Filter too short inference (because for frequency experiments, the intensity threshold is not very good)
    latency = dataframe.groupby("Inference_number")["Time (s)"].max() - dataframe.groupby("Inference_number")["Time (s)"].min()
    good_number = latency[latency > time_threshold].index.values.tolist()
    dataframe = dataframe[dataframe["Inference_number"].isin(good_number)]
    
    return dataframe[["Time (s)", "Inference_number", "Intensity (A)", "Power (W)"]]
    
def Get_MeanLatency_And_MeanEnergy(dataframe):
    
    # Select 50 inferences except the first one
    inference_list = dataframe["Inference_number"].unique()
    dataframe = dataframe[dataframe["Inference_number"].isin(inference_list[1:51])]
    
    # Integrate the power for each inference to get energy cunsumption and average over 50 inferences
    consumption = dataframe.groupby("Inference_number")["Power (W)"].sum()
    consumption = consumption*1e-4 # *1e-4 because their is 10000 samples by seconds.
    consumption = consumption.mean()
    
    # Compute latency and average over 50 inferences
    latency = dataframe.groupby("Inference_number")["Time (s)"].max() - dataframe.groupby("Inference_number")["Time (s)"].min()
    latency = latency.mean()
    
    return consumption,latency
    
def Get_Inference_AveragePower(dataframe):
    
    # Select 50 inferences except the first one
    inference_list = dataframe["Inference_number"].unique()
    dataframe = dataframe[dataframe["Inference_number"].isin(inference_list[1:51])]
    
    # Average power for each inference and average it over 50 inferences
    average_power = dataframe.groupby("Inference_number")["Power (W)"].mean()
    average_power = average_power.mean()
    
    return average_power