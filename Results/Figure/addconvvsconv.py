from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np

from analyse import *

intensity_threshold = 0.010
time_threshold = 0.00
parent_path = str(Path(os.getcwd()).parent.absolute())+os.sep+"Experiments"+os.sep+"Experiments_AddconvVsConv"

liste_folder = []
liste_folder.extend(glob.glob(parent_path+os.sep+"*/", recursive = True))

result_dataframe = pd.DataFrame(columns=['Name', 'Latency (s)'])
for i,folder in enumerate(liste_folder):
    row = []
    string = folder.split("AddconvVsConv/")[1].split("/")[0]
    row.append(string)
    dataframe = Unzip_Experiment_To_Dataframe(folder)
    dataframe = Get_Inference_In_Dataframe(dataframe, intensity_threshold, time_threshold)
    consumption, latency = Get_MeanLatency_And_MeanEnergy(dataframe)
    row.append(latency)
    print(row)
    result_dataframe.loc[i] = row
    
print(result_dataframe)

