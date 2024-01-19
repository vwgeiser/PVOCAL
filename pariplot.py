# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:02:23 2023

@author: vwgei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\PVOCAL_data_reduced.csv"

data = pd.read_csv(file_path)

datat0 = data.iloc[:,2:13]

datat24 = data.iloc[:,13:26]

dataVOC = data.iloc[:,34:134]

dataF = data.iloc[:, np.r_[13:26,34,39,75,89,118,119]]


# Plot Correlation coefficent using a heatmap
plt.figure(figsize=(20,18))
cor=dataF.corr(method='pearson')
cor = abs(cor)
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/CorrMatrixt-24_pearson.png")
plt.show()



# # Plot all features and target using a pairplot
# sns.pairplot(datat0, height=2.5)
# plt.tight_layout()
# plt.show()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotT0.png")

# Plot all features and target using a pairplot
# sns.pairplot(datat24, height=2.5)
# plt.tight_layout()
# plt.show()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotT24.png")

# # Plot all features and target using a pairplot
# sns.pairplot(dataVOC)
# plt.tight_layout()
# plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/PairPlotAllVOC.png")
# plt.show()