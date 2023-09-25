# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:24:11 2023

@author: Victor Geiser

This file uses PySPLIT to generate bulk trajectories for every starting canister.
One of the hardest parts of running this model is generating the trajectories accuratly.

This file is currently tailored to the final csv that I used which is attached
as a part of the GitHub repository. 

Have fun with this part. It is a lot of csv wrangling :)
It is expected that this aspect of any machine learning project will take 50-80% of a programmers time.

"""

import pandas as pd
import pysplit 

# The path to your CSV that has all of the requirements for "generate_bulktraj"
csvPath = r"C:\Users\vwgei\Documents\PVOCAL\data\PVOCALRFInput.csv"

# Read in data
df1 = pd.read_csv(csvPath)

# Get number of rows that have data
dataLength = len(df1)

# Read in Lats and Longs
lat = df1.iloc[:, 44] #10
long = df1.iloc[:, 45] #11

#lat-long tuples
LL = list(zip(lat,long))

# Index of dates/times in string format
timestampInd = 37

# Get the column that has our dates    
dates = df1.iloc[:,timestampInd]

# Insert our Lat/Long tuples into the dataframe
df1.insert(timestampInd+4, "LL", LL) 

# Get all of the times
times = df1.iloc[:,43]

# Container for times as a string
strHours = []

# Remove the hours and minutes bc file names need to not have ":" in them
for time in times:
    tempt = time.split(":")
    #tempt = ''.join(tempt)
    strHours.append(tempt[0])
    
# Add the hours of initial trajectory launch into our dataframe    
df1.insert(44, "HoursN", strHours)

# Directory containers for hysplit and where our output files will be
working_dir = 'C:/hysplit/working'
storage_dir = 'D:/PVOCAL/GDASTrajectories'
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
5. select all variables (i recommend all if computation time isn't a huge factor')
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
# Add timing and altitude to basename to create unique name # INDEX CHANGE HERE
trajname = (basename + str(iindex) + m_str + '{:04}'.format(a) + season + 
            fnameyr + "{0:02}{1:02}{2:02}".format(m, d, h))
# NOTE: notice once again the inclusion of the "str(iindex)"   
"""                
for ind in df1.index:
    #print(df1["OpenTime"][ind]) #DEBUG
    pysplit.generate_bulktraj(basename, 
                              working_dir, 
                              storage_dir, 
                              meteo_dir, 
                              years = [df1["YearN"][ind]], 
                              months = [df1["MonthN"][ind]], 
                              hours = [df1["HoursN"][ind]], 
                              altitudes = [df1["AltP_meters_0"][ind]], 
                              coordinates = df1["LL"][ind], 
                              run = (-24),
                              iindex = [df1["iindex"][ind]],
                              meteoyr_2digits = True,
                              outputyr_2digits = False,
                              monthslice = slice(df1["DayN"][ind], df1["DayN"][ind] + 1, 1),
                              meteo_bookends = ([4,5],[1]),
                              get_reverse = False,
                              get_clipped = False,
                              hysplit = "C:\hysplit\exec\hyts_std"
                              )
    