# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:12:48 2023

This is the main python file for the PVOCAL model. 

This file has several features: 
    1) Possible prediction of all VOCs in inital spreadshee (8 main variables listed)
    2) Prediction of VOC concentrations using both 0 hour and -24 hour trajectory locations
    3) Ability to do average error metric over any number of model runs
    4) Calculate feature importances for input features
    5) Can plot the resulting decision tree (currently commented out see "tree plotting" section)

@author: Victor Geiser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import d2_absolute_error_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
#from sklearn.metrics import accuracy_score

# These packages are used when the code to plot the decision tree is uncommented
from sklearn.tree import plot_tree
from sklearn.tree import export_text

# Read in CSV file with all of the data we need. Meteorology variables + Pathdata
file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\PVOCALRFInput.csv"

# Load data from CSV into a pandas DataFrame.
data = pd.read_csv(file_path)

# VOC variable for prediction
v0 = "DMS (MS)" 
v1 = "CH4 (ppmv)" 
v2 = "Benzene (E_MS)" 
v3 = "Toluene (B)" 
v4 = "Isoprene (E_B)"
v5 = "Ethane (E)"
v6 = "H-1211 (C_D)" # non-operational
v7 = "H-1301 (C)"  # non-operational

# Time variable for prediction 
t0 = "T0"
t1 = "T-24"
#------------------------------------------------------------------------------


# Model Run Parameters
var_of_int = v0 # Must be v0-7
time = t0 # Either t0 or t1
numRuns = 1 # Must be greater than or equal to 1


#------------------------------------------------------------------------------
# Gets rid of bad values in the predicted variable
data = data.dropna(subset = var_of_int)

# Set input variables for model T0 endpoints vs T-24 endpoints
if (time == "T0"):
    predictionBaseVars = data.iloc[:, 3:14] # Initial point data at time: 0 hr
else:
    predictionBaseVars = data.iloc[:, 16:32] # Backwards trajctory point data at time: -24hr # with distance features

# Get the VOC of interest
prediction_var = data.loc[:, [var_of_int]]

# Initialize arrays to hold run by run averages
r2arr = []
#accArr =[]
d2arr = []
MSEarr = []
EVSarr = []
maxErrarr = []
oobarr = []
FeatImportances = []
STDImportances= []


##
### This is the main algorithm of the file!
### Scikit-Learn does the heavy lifting of the machine learning
##
for i in range(numRuns):
    # Split the data into training and testing sets. # random_state=1911816
    x_train, x_test, y_train, y_test = train_test_split(predictionBaseVars, prediction_var, test_size=0.2)

    # Initialize the Random Forest Regressor
    """
    These model "hyperparameters" currently produce the best results for the model. There are a couple notable chages from the default RFR:
        1) criterion='absolute_error' - Although run time increases significantly feature importances are reported with lower error
            - Using absolute error minimized L1 loss
        2) oob_score=True - Calculates an out of box score. The fact that this is similar to our standard r^2 score is a good indicator that our model is not overfitting.
        3) max_depth=11 - This indicates a max decision tree depth of max_depth currently set at 11 as this is the minimum depth of the model that does not reduce r^2
        4) max_features='sqrt' - Takes the square root of input features for the decision tree prior to tree "voting"
        5) n_jobs=-1 - Enables the use of all prcosessors. Increases compututational power requriements but also increases runtime
        6) bootstrap=True - Trains each tree on a random sampling of a subset of observations. This is a complex topic so please refer to Scikit-Learn documentation to learn more: 
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        
    
    """
    
    runParameters = var_of_int + time
 
    ## It was observed that during hyperparameter finetuning that hyperparametes were not constant between VOCs so the following dictionaries were made for storing optimal hyperparameters for individual VOCs
    ## These values are derived from the "PVOCALhyperparameterTuning.py" file
    # Modifies the RF hyperparameter n_estimators
    nEST = {'DMS (MS)T-24':1000, 'Isoprene (E_B)T-24':1800, 'CH4 (ppmv)T-24':200, 'Benzene (E_MS)T-24':200, 'Toluene (B)T-24':200, 'Ethane (E)T-24':1200, 'DMS (MS)T0':200, 'Isoprene (E_B)T0':200, 'CH4 (ppmv)T0':800, 'Benzene (E_MS)T0':1800, 'Toluene (B)T0':200, 'Ethane (E)T0':200}
    # Modifies the RF hyperparameter min_samples_split
    minSampSplit = {'DMS (MS)T-24':2, 'Isoprene (E_B)T-24':5, 'CH4 (ppmv)T-24':2,'Benzene (E_MS)T-24':2, 'Toluene (B)T-24':2, 'Ethane (E)T-24':2, 'DMS (MS)T0':2, 'Isoprene (E_B)T0':2, 'CH4 (ppmv)T0':5, 'Benzene (E_MS)T0':5, 'Toluene (B)T0':10, 'Ethane (E)T0':10}
    # Modifies the RF hyperparameter min_samples_leaf
    minSampLeaf = {'DMS (MS)T-24':1, 'Isoprene (E_B)T-24':2, 'CH4 (ppmv)T-24':1,'Benzene (E_MS)T-24':1, 'Toluene (B)T-24':1, 'Ethane (E)T-24':1, 'DMS (MS)T0':1, 'Isoprene (E_B)T0':1, 'CH4 (ppmv)T0':1, 'Benzene (E_MS)T0':1, 'Toluene (B)T0':1, 'Ethane (E)T0':1}
    # Modifies the RF hyperparameter max_depth
    maxDepth = {'DMS (MS)T-24':110, 'Isoprene (E_B)T-24':60, 'CH4 (ppmv)T-24':110, 'Benzene (E_MS)T-24':110, 'Toluene (B)T-24':110, 'Ethane (E)T-24':110, 'DMS (MS)T0':80, 'Isoprene (E_B)T0':80, 'CH4 (ppmv)T0':70, 'Benzene (E_MS)T0':110, 'Toluene (B)T0':90, 'Ethane (E)T0':110}
    # Modifies the RF hyperparameter criterion
    errorMetric = 'absolute_error'
    # Modifies the RF hyperparameter max_features
    maxFeat = {'DMS (MS)T-24':'sqrt', 'Isoprene (E_B)T-24':'sqrt', 'CH4 (ppmv)T-24':'sqrt', 'Benzene (E_MS)T-24':'sqrt', 'Toluene (B)T-24':'sqrt', 'Ethane (E)T-24':'sqrt', 'DMS (MS)T0':'sqrt', 'Isoprene (E_B)T0':'sqrt', 'CH4 (ppmv)T0':'log2', 'Benzene (E_MS)T0':'log2', 'Toluene (B)T0':'log2', 'Ethane (E)T0':'sqrt'}
    # Modifies the RF hyperparameter verbose
    loud = 0
    # Modifies the RF hyperparameter oob_score
    outOfBox = True
    # Modifies the RF hyperparameter bootstrap
    boots = True
    
    # Initialize the Random Forest Regressor using Sci-Kit Learn
    rf_regressor = RandomForestRegressor(n_estimators=nEST[runParameters], min_samples_split=minSampSplit[runParameters], min_samples_leaf=minSampLeaf[runParameters], max_depth=maxDepth[runParameters], oob_score=outOfBox, criterion='absolute_error', max_features=maxFeat[runParameters], n_jobs=-1, verbose=loud, bootstrap=boots)
    
    # Train the regressor on the training data.
    rf_regressor.fit(x_train, y_train.values.ravel())

    # Make predictions on the test data.
    y_pred = rf_regressor.predict(x_test)

    ##
    ### Calculate SciKitLearn model scoring metrics.
    ##

    # How good of an overall fit is the model to the data, the coefficient of determination 
    r2arr.append(r2_score(y_test, y_pred))
    
    # Scikit-Learn Accuracy? -> Nope not for regression
    #accArr.append(accuracy_score(y_test, y_pred))

    # Generalization of r2 (skill score) squared error replaced by deviance mean absolute error
    d2arr.append(d2_absolute_error_score(y_test, y_pred))

    # Average Squared Error for the model
    MSEarr.append(mean_squared_error(y_test, y_pred))

    # Explained variance score does not account for systematic offset in the prediction
    # Predictor is not biased (not bias if EXV=R2)
    EVSarr.append(explained_variance_score(y_test, y_pred))

    # Largest Error between prediction and observed value
    maxErrarr.append(max_error(y_test, y_pred))
    
    # Average out of box score
    oobarr.append(rf_regressor.oob_score_)
    
    
    # ------------------------------ Sensitivity analysis --------------------
    # Calculate importances
    importances = rf_regressor.feature_importances_
    FeatImportances.append(importances)

    # Calculate Standard Deviation along axis
    std = np.std([tree.feature_importances_ for tree in rf_regressor.estimators_], axis=0)
    STDImportances.append(std)

# -----------------------------------------------------------------------------
# This section of code is not yet operational - supposed to calculate P value

# lm = LinearRegression()
# lm.fit(X,y)
# params = np.append(lm.intercept_,lm.coef_)
# predictions = lm.predict(X)
# new_X = np.append(np.ones((len(X),1)), X, axis=1)
# M_S_E = (sum((y-predictions)**2))/(len(new_X)-len(new_X[0]))
# v_b = M_S_E*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
# s_b = np.sqrt(v_b)
# t_b = params/ s_b
# p_val =[2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
# p_val = np.round(p_val,3)
# p_val

# -----------------------------------------------------------------------------

# Calculate Averages based on run data
r2ave = np.average(r2arr)
#accave = np.average(accArr)
d2ave = np.average(d2arr)
MSEave = np.average(MSEarr) 
EVSave = np.average(EVSarr)
maxErrave = np.average(maxErrarr)
oobave = np.average(oobarr)

# This code segment effects the feature importances graph. It calculates the average feature importance for the "numRuns" 
if numRuns > 1:   
    # Calculate the average feature importances
    FinalAvgedImportances = []
    FinalAvgedSTD = []
    # i is the number of input features
    for i in range(len(FeatImportances[0])):
        TotalImportance = []
        TotalSTD = []
        
        # j is the number of runs
        for j in range(len(FeatImportances) - 1):
            TotalImportance.append(FeatImportances[j][i]) # warning here
            TotalSTD.append(STDImportances[j][i])
        
        # Calculate and put averages into container
        AveIndImportance = np.average(TotalImportance)
        AveIndSTD = np.average(TotalSTD)
        FinalAvgedImportances.append(AveIndImportance)
        FinalAvgedSTD.append(AveIndSTD)
        
  
# Print Evaluation metrics to console
print(f"Number of Model Runs: {numRuns}")
print(f"R2 Average: {r2ave:.2f}")
#print(f"Accuracy Score: {accave:.2f}")  
print(f"D2 Average: {d2ave:.2f}")
print(f"Mean Squared Error Average: {MSEave:.0f}")
print(f"Explained Variance Score Average: {EVSave:.3f}") 
print(f"Max Error Average: {maxErrave:.0f}")
print(f"Out of Box Score Average: {oobave:.3}")

# Calculate manuel error variables
def percentErr(true, predicted):
    predictionErr = abs((true - predicted) / true) * 100
    return predictionErr


# Format CSV column name for predicted variable
observed = 'observed'
obs_label = observed + var_of_int
predicted = 'predicted'
predic_label = predicted + var_of_int

# Make path for output .csv
filebase = 'C:/Users/vwgei/Documents/PVOCAL/data/'
extension = '.csv'
predPath = filebase + "predicted" + var_of_int + extension

# Initiallize output dataframe
PredDF = pd.DataFrame()

# Write in data that was used for predictions
PredDF = x_test

# Write in data "correct" prediction values
PredDF[obs_label] = y_test

# Write in predicted values 
PredDF[predic_label] = y_pred

# Add Percent Error calculation rounded
PredDF["Percent_Error"] = round(percentErr(PredDF[obs_label], PredDF[predic_label]), 2)

# Write it to a .csv
PredDF.to_csv(predPath)


#------------------------------- Tree Plotting --------------------------------
# # Visualize decision tree - Increases run time significantly
# fig = plt.figure(figsize=(30, 20))
# plot_tree(rf_regressor.estimators_[0], 
#           filled=True, impurity=True, 
#           rounded=True)
#------------------------------------------------------------------------------

## Plot feature importances

# -24hr variable labels
labels24 = ['Pressure', 'Theta Temp', 'Rainfall',
        'Mixing Depth', 'Specific Humidity', 'Terrain Altitude',
        'Solar Radiation', 'Latitude', 'Longitude',
        'Altitude', 'Temperature', 'Distance ptp',
        'Cumulative Dist', 'Dist from origin',
        'Bearing from origin', 'Bearing ptp']

# 0hr labels
labels0 = ['Pressure', 'Theta Temp', 'Rainfall', 'Mixing_Depth',
        'Specific Humidity', 'Terrain Altitude', 'Solar Radiation',
        'Latitude', 'Longitude', 'Altitude', 'Temperature']

# Initiallize empy array to store cleaned labels for display purposes
finalLabels = []
timeParameter = ""

# Get the labels that corresponsd to the input variables on the excel sheet
if (len(predictionBaseVars.columns) < 15):
    finalLabels = labels0
    timeParameter = " T0"
else:
    finalLabels = labels24
    timeParameter = " T-24"

if numRuns > 1:
    # Match data to data labels
    forest_importances = pd.Series(FinalAvgedImportances, index=finalLabels)
else:
    forest_importances = pd.Series(importances, index=finalLabels)

# Get just the VOC name
var_of_int = str.split(var_of_int)[0]

# Change the label of CH4 to methane for visual appearance
if (var_of_int == "CH4"):
    var_of_int = "Methane"

# Plot feature importances
fig, ax = plt.subplots()
# Plot as histogram

if numRuns > 1:
    forest_importances.plot.bar(yerr=FinalAvgedSTD, ax=ax) #FinalAvgedSTD #std
else:
    forest_importances.plot.bar(yerr=std, ax=ax)

# Axis label
ax.set_title("Feature importances using MDI for " + var_of_int + timeParameter)
ax.set_ylabel("Mean decrease in impurity")
# Format Graph
fig.tight_layout()

#------------------------------------------------------------------------------