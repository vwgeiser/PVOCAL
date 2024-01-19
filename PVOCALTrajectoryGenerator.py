# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:19:17 2023

@author: vwgei
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pysplit 

# Read in data
data = pd.read_csv(r"C:\Users\vwgei\Downloads\Sarp_Airborne_2009_2022.csv")

# Replace values below WAS limit of dectection with np.NaN for consistancy
# data.replace(0.001, np.nan, inplace=True)
# Replace -888 values with np.NaN for consistancy
# data.replace(-888, np.nan, inplace=True)


data = data.dropna(subset = "Latitude")
data = data.dropna(subset = "Longitude")
data = data.dropna(subset="AltP_feet")

data = data[data['AltP_meters'] < 10000]

dataLength = len(data)

# Read in Lats and Longs
lat = data.iloc[:, 5] #10
long = data.iloc[:, 6] #11

#lat-long tuples
# LL = list(zip(lat,long))
data['LL'] = list(zip(lat,long))

# Assuming df is your DataFrame and DateTime is the column containing datetime information
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Create new columns
data['Year'] = data['DateTime'].dt.year
data['Month'] = data['DateTime'].dt.month
data['Day'] = data['DateTime'].dt.day
data['Hour'] = data['DateTime'].dt.hour



# Directory containers for hysplit and where our output files will be
working_dir = 'C:/hysplit/working'
storage_dir = 'C:/Users/vwgei/Documents/PVOCAL/data/GDASTrajectories'
meteo_dir = 'D:/PVOCAL/GDAS'

# Basename shared by all trajectories
basename = 'PVOCALGDAS'
                  
"""
*****NOTE*****
HYSPLIT is capable of calculating meteorlogical variables along the trajctory 
path. I recommend you do this for this model. To do this after you have 
HYSPLIT on your local machine: 
    
THIS REQUIRES A LOCAL INSTALLATION OF HYSPLIT (either registered or unregistered version - the unregistered version was used for the inital project)

1. Run HySPLIT
2. click "Menu"
3. click "Advanced"->"Configuration settings"->"Trajectory"
4. clik "Menu" FOR (6) "Add METEOROLOGY output along trajectory"
5. select all variables (all is recommended if computation time isn't a huge factor')
6. CLICK SAVE
7. CLICK SAVE AGAIN
8. File size for each trajectory should be about 5 kilobites as opposed to 3 (for 24 hour back trajecories)
"""

# Do this ^^^ (if it makes sense for you...)

"""
This is the main function PySPLIT is used for. 

There is however one difference between this function and the generate_bulktraj
that comes with pysplit as of pysplit version=0.3.5

This is the inclusion of the iindex parameter is special for this program and 
my model in that the data I was working with had no index or unique ID to match
with accross multiple iterations of excel joining and other shenanigans.

This might not be necessary for your project but it was for mine as some entire
rows needed to be thrown out.

There could also be some way that this is done in PySPLIT or a better way
to generate a usible index not in this form but this is the best way that I 
figured out for the time/problems I was having.

...

The changes to trajectory_generator.py within pysplit are as follows:    
    
# Change at line 11 in trajectory_generator.py
def generate_bulktraj(basename, hysplit_working, output_dir, meteo_dir, years,
                      months, hours, altitudes, coordinates, run, iindex,
                      meteoyr_2digits=True, outputyr_2digits=False,
                      monthslice=slice(0, 32, 1), meteo_bookends=([4, 5], [1]),
                      get_reverse=False, get_clipped=False,
                      hysplit="C:\\hysplit4\\exec\\hyts_std"):
# NOTE: notice the new "iindex" parameter


# Change at line 150 to trajectory_generator.py    
# Add trajectory index and altitude to basename to create unique name # INDEX CHANGE HERE
trajname = (basename + str(iindex) + m_str  + season + 
            fnameyr + "{0:02}{1:02}{2:02}".format(m, d, h))
# NOTE: notice once again the inclusion of the "str(iindex)"   
"""

data.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\TrajectoryGeneratorInput.csv",index=False)
                
for ind in data.index:
    # print(data["DateTime"][ind]) #DEBUG
    pysplit.generate_bulktraj(basename, 
                              working_dir, 
                              storage_dir, 
                              meteo_dir, 
                              years = [data["Year"][ind]], 
                              months = [data["Month"][ind]], 
                              hours = [data["Hour"][ind]], 
                              altitudes = [data["AltP_meters"][ind]], 
                              coordinates = data["LL"][ind], 
                              run = (-24),
                              iindex = [data["iindex"][ind]],
                              meteoyr_2digits = True,
                              outputyr_2digits = False,
                              monthslice = slice(data["Day"][ind] - 1, data["Day"][ind], 1),
                              meteo_bookends = ([4,5],[1]),
                              get_reverse = False,
                              get_clipped = False,
                              hysplit = "C:\hysplit\exec\hyts_std"
                              )

