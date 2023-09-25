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


# Make trajectory group #PySPLIT function
trajgroup = pysplit.make_trajectorygroup(filelist)

# Intialize pd dataframes as containers
dfF = pd.DataFrame()
dfL = pd.DataFrame()
df1 = pd.DataFrame()

# Get the list of all files and directories
path = 'D:/PVOCAL/GDASTrajectories' #"C:/Users/SARP/Documents/NASASARP/trajectories"
dir_list = os.listdir(path)

# Funtion that extracts the custom "iindex" for the file names 
def extract_between_regex(text, start, end):
    pattern = rf"{re.escape(start)}(.*?){re.escape(end)}"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""

# Initialize containers for string and int indecies
final = []
finalInt = []

# Utilize the above function to pull the index from the file name and put it into a list we can use
for file in dir_list:
    x = extract_between_regex(file, "[", "]")
    final.append(x)

# This could probably be condesed with the above loop but this just makes it an integer
for item in final:
    Intitem = int(item)
    finalInt.append(Intitem)
    
# Initialize a manual index
mi = 0

# Get the created "iindex" feature from the data frame that has been matched up to the correct trajectories.
intFinalDF = pd.DataFrame(finalInt, columns=['iindex'])

# -----------------------------------------------------------------------------
# An experimental test to import trajectory shapely lines as a shapefile 
# It did not work the last time that I ran it but I have left it in as I see it being useful in the future
#filebase = "C:/Users/SARP/Documents/NASASARP/data/shapefiles/"
#extention = '.shp'


# for traj in trajgroup:
#     x = traj.data
#     x.drop(columns="Timestep",inplace=True)
#     x.drop(columns="DateTime",inplace=True)
    
#     filename = final[mi]

#     fullpath = filebase + filename + extention
        
#     x.to_file(fullpath,crs='EPSG:4326')
    
#     mi = mi + 1
# -----------------------------------------------------------------------------

# Reset the manual index if the shapefiles were made
#mi = 0

"""
HYSPLIT is already capable of calculating meteorological variables along the trajectory path however part of my future work will be to integrate the entirety of this HYSPLIT trajectoy path into the model. 

I also see an avenue for future work to include the fundementals of atmospheric photochemistry modeling in order to make better predictions.
"""
# get final pathdata endpoints from the 0 hour and -24 hour timestamps
for traj in trajgroup:
    #df1 = df1.append(trajgroup[mi].data.geometry) #DEBUG - unsused
    traj.calculate_distance()
    traj.calculate_vector()
    testAlt = traj.data.iloc[0][9]
    #testtest.append(testAlt) # DEBUG
    x = traj.data.iloc[0]
    x = x.append(intFinalDF.iloc[mi])
    dfF = dfF.append(x,ignore_index=True)
    y = traj.data.iloc[-1]
    y = y.append(intFinalDF.iloc[mi])
    dfL = dfL.append(y,ignore_index=True)
    mi = mi + 1
#     #df1 = df1.append(traj.data.iloc[0]) #DEBUG - unused
    
# [Not operational] was supposed to generate shapefiles
#traj.data.to_file(r"C:/Users/SARP/Documents/NASASARP/data/shapefiles",crs="EPSG:4326") 

# Export data frames to csv
dfF.to_csv(r"C:\Users\SARP\Documents\NASASARP\data\trajdata0.csv")
dfL.to_csv(r"C:\Users\SARP\Documents\NASASARP\data\trajdata-24.csv")

# [Not fully operational] It follows that the same process above could be used on HYSPLIT reverse back trajectories as well.    
# for traj in trajgroup:
#     traj.load_reversetraj()
    