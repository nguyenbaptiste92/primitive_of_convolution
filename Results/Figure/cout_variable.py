import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

from sklearn.linear_model import LinearRegression

func = (lambda a,b,c,d,g : 1/2*c/g*(20*d**2*b/g+67)+26)

liste = [i for i in range(1,33)]
variable_cost = [func(32,16,16,3,g)-26 for g in liste]
fixed_cost= [26 for i in range(1,33)]
fixed_variable_cost = [i/j for i,j in zip(fixed_cost,variable_cost)]

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(4, 2))

ax.plot(liste, variable_cost, color = 'c', label='Cout variable')
ax.plot(liste,fixed_cost, color = 'r', label='Cout fixe')
ax.legend()
ax.set_xlabel('Groupes', fontsize=10)
ax.set_ylabel("Nombre d'instructions", fontsize=10)
ax.set_yscale('log') 
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'cout.jpeg',bbox_inches='tight',pad_inches = 0, format='jpeg')
fig.savefig(os.getcwd()+os.sep+"Figure"+os.sep+'cout.eps',bbox_inches='tight',pad_inches = 0, format='eps')
