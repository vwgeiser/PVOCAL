# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 19:16:26 2023

This file's purpose is to calculate distance and vector data along the trajectory path.
The trajectories used in this file should already have meteorology data with them.

See the "***NOTE***" within the file "PVOCALTrajectoryGenerator" if this is not the case.

Additionaly this file utilizes the one change to the actual PySPLIT "trajectory_generator" file in the need for an iindex. This is becuase during initial trajectory generation there was a need ensure the correct data for trajectories is matched up with the correct pathdata. Since "altitude" is represented in the file name of a trajectory by PySPLIT default the index variable "iindex" was an way to ensurue that all trajectoies needed to have a unique feature to join/manipulate.

@author: Victor Geiser
"""

from __future__ import print_function

import pysplit
import pandas as pd
# import geopandas as gpd
#import shapely as sp #DEBUG - no longer used 
import os
import re

"""
There is potential for this file to eventually store all of the pathdata from the trajectories using xarray. Integrating this into the file would mean there would be a challenge representing this in the output excel. This is an aspect of this model I would lke to incoperate in the future.

#import xarray as xr 
"""


# Get the list of all files and directories
path = 'D:/PVOCAL/GDASTrajectories' #"C:/Users/SARP/Documents/NASASARP/trajectories"
dir_list = os.listdir(path)

# Initialize a contaner for the file list
filelist = []

basename = 'D:/PVOCAL/GDASTrajectories/' #"C:/Users/SARP/Documents/NASASARP/trajectories/"

# Create a list of file names concatenating the file base path to the trajectory name
for file in dir_list:
    tmpStr = basename + file
    filelist.append(tmpStr)


# sorted_file_paths = sorted(filelist, key=lambda x: int(x.split('[')[1].split(']')[0]))

# Make trajectory group #PySPLIT function
# trajgroup = pysplit.make_trajectorygroup('D:/PVOCAL/GDASTrajectories/*')
trajgroup = pysplit.make_trajectorygroup(filelist)

# Intialize pd dataframes as containers
dfF = pd.DataFrame()
dfL = pd.DataFrame()
df1 = pd.DataFrame()

# Get the list of all files and directories
path = 'D:/PVOCAL/GDASTrajectories' #"C:/Users/SARP/Documents/NASASARP/trajectories"
dir_list = os.listdir(path)

#sort them in a way that makes sense

# Funtion that extracts the custom "iindex" for the file names 
def extract_between_regex(text, start, end):
    pattern = rf"{re.escape(start)}(.*?){re.escape(end)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""

# # Initialize containers for string and int indecies
# final = []
# finalInt = []

# # Utilize the above function to pull the index from the file name and put it into a list we can use
# for file in dir_list:
#     x = extract_between_regex(file, "[", "]")
#     final.append(x)

# # This could probably be condensed with the above loop but this just makes it an integer
# for item in final:
#     Intitem = int(item)
#     finalInt.append(Intitem)
    
# Initialize a manual index
mi = 0

# # Get the created "iindex" feature from the data frame that has been matched up to the correct trajectories.
# intFinalDF = pd.DataFrame(finalInt, columns=['iindex'])

# # Convert the NumPy array to a Python list of lists
# sorted_intFinalDF = intFinalDF.sort_values(by='iindex')


# Reset the manual index if the shapefiles were made
# mi = 0
column_names = [
    'Timestep',
    'Pressure',
    'Potential_Temperature',
    'Temperature',
    'Rainfall',
    'Mixing_Depth',
    'Relative_Humidity',
    'Specific_Humidity',
    'Mixing_Ratio',
    'Terrain_Altitude',
    'Solar_Radiation',
    'geometry',
    'DateTime',
    'Temperature_C',
    'Distance_ptp',
    'Cumulative_Dist',
    'Dist_from_origin',
    'bearings_from_origin',
    'bearings_ptp',
    'iindex'
]


"""
HYSPLIT is already capable of calculating meteorological variables along the trajectory path however part of my future work will be to integrate the entirety of this HYSPLIT trajectoy path into the model. 

I also see an avenue for future work to include the fundementals of atmospheric photochemistry modeling in order to make better predictions.
"""
dfF = pd.DataFrame()
dfL = pd.DataFrame()

dfF = []
dfL = []

# dfF = trajgroup[0].data.columns
# dfF = trajgroup[0].data.columns
# get final pathdata endpoints from the 0 hour and -24 hour timestamps
for traj in trajgroup:
    #df1 = df1.append(trajgroup[mi].data.geometry) #DEBUG - unsused
    traj.calculate_distance()
    traj.calculate_vector()
    # traj.calculate_moistureflux()

    x = traj.data.iloc[0]    
    
    
    x.loc['iindex'] = extract_between_regex(traj.filename, "[", "]")
    
    x = x.to_list()
    
    
    dfF.append(x)
    
    # dfF = pd.concat([dfF, x], ignore_index=True)
    # pd.concat([dfF,x.to_frame().T],ignore_index=True)
    
    y = traj.data.iloc[-1]
    
    # number_to_insert = intFinalDF[mi]
    y.loc['iindex'] = extract_between_regex(traj.filename, "[", "]")
    
    y = y.to_list()
    
    dfL.append(y)
    
    mi = mi + 1

df1 = pd.DataFrame(dfF, columns=column_names)
df2 = pd.DataFrame(dfL, columns=column_names)

# Export data frames to csv
df1.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\CSVFun\trajdata0.csv", index=False)
df2.to_csv(r"C:\Users\vwgei\Documents\PVOCAL\data\CSVFun\trajdata-24.csv", index=False)

# [Not fully operational] It follows that the same process above could be used on HYSPLIT reverse back trajectories as well.    
# for traj in trajgroup:
#     traj.load_reversetraj()
    