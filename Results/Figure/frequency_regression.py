import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

csv_file = os.getcwd()+os.sep+"Dataframe"+os.sep+"Frequency_Latency_And_Energy.csv"
dataframe = pd.read_csv(csv_file,sep=",")
dataframe["Frequency"]=  dataframe["Frequency (MHz)"]*1e6
dataframe["inverse_Frequency"] = 1/dataframe["Frequency"]

dsp_dataframe = dataframe[(dataframe["DSP"]==1) & (dataframe["Layer type"]=="Conv")].sort_values(by=['Frequency (MHz)'], ascending=True)
no_dsp_dataframe = dataframe[(dataframe["DSP"]==0) & (dataframe["Layer type"]=="Conv")].sort_values(by=['Frequency (MHz)'], ascending=True)
print(dsp_dataframe)
print(no_dsp_dataframe)

frequency = [10*1e6,20*1e6,40*1e6,80*1e6]
naive_power = [16.16*1e3,21.59*1e3,32.83*1e3,52.09*1e3]
simd_power = [17.57*1e3,24.66*1e3,37.33*1e3,62.75*1e3]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(np.array(frequency).reshape(-1, 1))

##################################################################################################################################################################
# No SIMD instructions
##################################################################################################################################################################
print("No SIMD : Frequency to Power")
reg1 = LinearRegression().fit(np.array(frequency).reshape(-1, 1), np.array(naive_power).reshape(-1, 1))
print(reg1.score(np.array(frequency).reshape(-1, 1), np.array(naive_power).reshape(-1, 1)))
print(reg1.coef_)
print(reg1.intercept_)

print("No SIMD : Inverse frequency to Latency")
reg2 = LinearRegression().fit(np.array(no_dsp_dataframe["inverse_Frequency"]).reshape(-1, 1), np.array(no_dsp_dataframe["Latency (s)"]).reshape(-1, 1))
print(reg2.score(np.array(no_dsp_dataframe["inverse_Frequency"]).reshape(-1, 1), np.array(no_dsp_dataframe["Latency (s)"]).reshape(-1, 1)))
print(reg2.coef_)
print(reg2.intercept_)

##################################################################################################################################################################
# SIMD instructions
##################################################################################################################################################################
print("SIMD : Frequency to Power")
reg3 = LinearRegression().fit(np.array(frequency).reshape(-1, 1), np.array(simd_power).reshape(-1, 1))
reg3.score(np.array(frequency).reshape(-1, 1), np.array(simd_power).reshape(-1, 1))
print(reg3.coef_)
print(reg3.intercept_)

print("SIMD : Inverse frequency to Latency")
reg4 = LinearRegression().fit(np.array(dsp_dataframe["inverse_Frequency"]).reshape(-1, 1), np.array(dsp_dataframe["Latency (s)"]).reshape(-1, 1))
reg4.score(np.array(dsp_dataframe["inverse_Frequency"]).reshape(-1, 1), np.array(dsp_dataframe["Latency (s)"]).reshape(-1, 1))
print(reg4.coef_)
print(reg4.intercept_)

"""power_1 = (lambda f : reg1.coef_[0][0]*(f**2)+reg1.coef_[0][1]*f+reg1.intercept_[0])
latence_1 = (lambda f : reg2.coef_[0][0]/f)
#func_2 = (lambda f : (6.76906452e-10*f**2-3.77956989e-19*f+0.01099667)*(9142576.69565217*(1/f)-0.00061278))

liste_range = [i for i in range(1,2000,1)]
naive_power = [power_1(i) for i in liste_range]
naive_latence = [latence_1(i) for i in liste_range]
naive_energy = [latence_1(i)*power_1(i) for i in liste_range]
#simd_conso = [func_2(i) for i in liste_range]

plt.plot(liste_range,naive_energy)
plt.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'test.jpeg', bbox_inches='tight',pad_inches = 0, format='jpeg')

print(np.argmin(naive_energy))
#print(np.argmin(simd_conso))"""
