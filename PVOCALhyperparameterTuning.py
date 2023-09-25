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

# Read in CSV file with all of the data we need. Meteorology variables + Pathdata
file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\PVOCALRFInput.csv"

# Load data from CSV into a pandas DataFrame.
data = pd.read_csv(file_path)

# Prediction variables
v0 = "DMS (MS)" 
v1 = "CH4 (ppmv)" 
v2 = "Benzene (E_MS)" 
v3 = "Toluene (B)" 
v4 = "Isoprene (E_B)"
v5 = "H-1211 (C_D)" 
v6 = "H-1301 (C)" 
v7 = "Ethane (E)"
var_of_int = v4

# Gets rid of bad values in the predicted variable
data = data.dropna(subset = var_of_int)

# Set input variables for model
predictionBaseVars = data.iloc[:, 3:14] # Initial point data at time: 0 hr
#predictionBaseVars = data.iloc[:, 16:32] # Backwards trajctory point data at time: -24hr

# Get the VOC of interest
prediction_var = data.loc[:, [var_of_int]]

# Initialize arrays to hold run by run averages
r2arr = []
d2arr = []
MSEarr = []
EVSarr = []
maxErrarr = []
oobarr = []
FeatImportances = []
STDImportances= []

##
### Scikit-Learn does the heavy lifting of the machine learning
##

#enter the number of runs desired # ***MUST BE GREATER THAN OR EQUAL TO 1***
numRuns = 1
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
    # rf_regressor = RandomForestRegressor(oob_score=True, max_depth=11, criterion='absolute_error', max_features='sqrt', n_jobs=-1, bootstrap=True) #n_estimators, max_depth, verbose criterion='absolute_error' 'friedman_mse' 'friedman_mse' 'poisson' 'squared_error' #max_features='sqrt' # Examples of other possible modifications
    #rf_regressor = RandomForestRegressor(random_state=42)
    
    # from sklearn.model_selection import GridSearchCV
    
    # full_pipline = Pipeline([
    #     ("preprocessing", preprocessing),
    #     ("random_forest", RandomForestRegressor(random_state=42))
    #     ])
    # param_grid([
        
    #     ])
    
    
    #https://github.com/WillKoehrsen/Machine-Learning-Projects/tree/master/random_forest_explained
    
    
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 5, 7, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    error_criteria = ["absolute_error", "squared_error"]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'criterion': error_criteria,
                   'bootstrap': bootstrap}
    #print(random_grid)
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 2)
    # Fit the random search model
    rf_random.fit(x_train, y_train.values.ravel())
    
    #print(rf_random.best_params_)
        
    # # Train the regressor on the training data.
    # rf_regressor.fit(x_train, y_train.values.ravel())

    # # Make predictions on the test data.
    # y_pred = rf_regressor.predict(x_test)

    # ##
    # ### Calculate SciKitLearn model scoring metrics.
    # ##

    # # How good of an overall fit is the model to the data, the coefficient of determination 
    # r2arr.append(r2_score(y_test, y_pred))

    # # Generalization of r2 (skill score) squared error replaced by deviance mean absolute error
    # d2arr.append(d2_absolute_error_score(y_test, y_pred))

    # # Average Squared Error for the model
    # MSEarr.append(mean_squared_error(y_test, y_pred))

    # # Explained variance score does not account for systematic offset in the prediction
    # # Predictor is not biased (not bias if EXV=R2)
    # EVSarr.append(explained_variance_score(y_test, y_pred))

    # # Largest Error between prediction and observed value
    # maxErrarr.append(max_error(y_test, y_pred))
    
    # # Average out of box score
    # oobarr.append(rf_regressor.oob_score_)
    
    
    # ------------------------------ Sensitivity analysis --------------------
    # # Calculate importances
    # importances = rf_regressor.feature_importances_
    # FeatImportances.append(importances)

    # # Calculate Standard Deviation along axis
    # std = np.std([tree.feature_importances_ for tree in rf_regressor.estimators_], axis=0)
    # STDImportances.append(std)

# -----------------------------------------------------------------------------
# This section of code is not yet operational 

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

# # Calculate Averages based on run data
# r2ave = np.average(r2arr)
# d2ave = np.average(d2arr)
# MSEave = np.average(MSEarr) 
# EVSave = np.average(EVSarr)
# maxErrave = np.average(maxErrarr)
# oobave = np.average(oobarr)

# if numRuns > 1:
   
#     # Calculate the average feature importances
#     FinalAvgedImportances = []
#     FinalAvgedSTD = []
#     # i is the number of input features
#     for i in range(len(FeatImportances[0])):
#         TotalImportance = []
#         TotalSTD = []
        
#         # j is the number of runs
#         for j in range(len(FeatImportances) - 1):
#             TotalImportance.append(FeatImportances[j][i]) # warning here
#             TotalSTD.append(STDImportances[j][i])
        
#         # Calculate and put averages into container
#         AveIndImportance = np.average(TotalImportance)
#         AveIndSTD = np.average(TotalSTD)
#         FinalAvgedImportances.append(AveIndImportance)
#         FinalAvgedSTD.append(AveIndSTD)
        
  
# # Print Evaluation metrics to console
# print(f"Number of Model Runs: {numRuns}")
# print(f"R2 Average: {r2ave:.2f}") 
# print(f"D2 Average: {d2ave:.2f}")
# print(f"Mean Squared Error Average: {MSEave:.0f}")
# print(f"Explained Variance Score Average: {EVSave:.3f}") 
# print(f"Max Error Average: {maxErrave:.0f}")
# print(f"Out of Box Score Average: {oobave:.3}")

# # Calculate manuel error variables
# def percentErr(true, predicted):
#     predictionErr = abs((true - predicted) / true) * 100
#     return predictionErr


# # Format CSV column name for predicted variable
# observed = 'observed'
# obs_label = observed + var_of_int
# predicted = 'predicted'
# predic_label = predicted + var_of_int

# # Make path for output .csv
# filebase = 'C:/Users/vwgei/Documents/PVOCAL/data/'
# extension = '.csv'
# predPath = filebase + "predicted" + var_of_int + extension

# # Initiallize output dataframe
# PredDF = pd.DataFrame()

# # Write in data that was used for predictions
# PredDF = x_test

# # Write in data "correct" prediction values
# PredDF[obs_label] = y_test

# # Write in predicted values 
# PredDF[predic_label] = y_pred

# # Add Percent Error calculation rounded
# PredDF["Percent_Error"] = round(percentErr(PredDF[obs_label], PredDF[predic_label]), 2)

# # Write it to a .csv
# PredDF.to_csv(predPath)


#------------------------------- Tree Plotting --------------------------------
# # Visualize decision tree - Increases run time significantly
# fig = plt.figure(figsize=(15, 10))
# plot_tree(rf_regressor.estimators_[0], 
#           filled=True, impurity=True, 
#           rounded=True)
#------------------------------------------------------------------------------

### Plot feature importances

# # -24hr variable labels
# labels24 = ['Pressure', 'Theta Temp', 'Rainfall',
#         'Mixing Depth', 'Specific Humidity', 'Terrain Altitude',
#         'Solar Radiation', 'Latitude', 'Longitude',
#         'Altitude', 'Temperature', 'Distance ptp',
#         'Cumulative Dist', 'Dist from origin',
#         'Bearing from origin', 'Bearing ptp']

# # 0hr labels
# labels0 = ['Pressure', 'Theta Temp', 'Rainfall', 'Mixing_Depth',
#         'Specific Humidity', 'Terrain Altitude', 'Solar Radiation',
#         'Latitude', 'Longitude', 'Altitude', 'Temperature']

# # Initiallize empy array to store cleaned labels for display purposes
# finalLabels = []
# timeParameter = ""

# # Get the labels that corresponsd to the input variables on the excel sheet
# if (len(predictionBaseVars.columns) < 15):
#     finalLabels = labels0
#     timeParameter = " T0"
# else:
#     finalLabels = labels24
#     timeParameter = " T-24"

# if numRuns > 1:
#     # Match data to data labels
#     forest_importances = pd.Series(FinalAvgedImportances, index=finalLabels)
# else:
#     forest_importances = pd.Series(importances, index=finalLabels)

# # Get just the VOC name
# var_of_int = str.split(var_of_int)[0]

# # Change the label of CH4 to methane for visual appearance
# if (var_of_int == "CH4"):
#     var_of_int = "Methane"

# # Plot feature importances
# fig, ax = plt.subplots()
# # Plot as histogram

# if numRuns > 1:
#     forest_importances.plot.bar(yerr=FinalAvgedSTD, ax=ax) #FinalAvgedSTD #std
# else:
#     forest_importances.plot.bar(yerr=std, ax=ax)

# # Axis label
# ax.set_title("Feature importances using MDI for " + var_of_int + timeParameter)
# ax.set_ylabel("Mean decrease in impurity")
# # Format Graph
# fig.tight_layout()

#------------------------------------------------------------------------------