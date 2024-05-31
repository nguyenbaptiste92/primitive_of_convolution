from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np

from analyse import *

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiments_Addconvfused"

print("Without fused")
dataframe = Unzip_Experiment_To_Dataframe(parent_path+os.sep+"Without_Fused")
dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
print("latency : ",latency)

print("With fused")
dataframe = Unzip_Experiment_To_Dataframe(parent_path+os.sep+"Fused")
dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
print("latency : ",latency)

    