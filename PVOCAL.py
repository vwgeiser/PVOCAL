# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:12:48 2023

This is the main python file for the PVOCAL model. 

This file has several features that can be controlled in the : 
    1) Predictions with and Without the use of SMOGN algorithm
    2) Prediction of VOC concentrations using both 0 hour and -24 hour trajectory locations
    3) Calculate feature importances for input features
    4) Can plot an example decision tree 

@author: Victor Geiser

Author note: Hey there! Thanks for checking out my project. Sorry for the spagetti code, I am in no way a software designer! Feel free to contact me if you have any questions

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import d2_absolute_error_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
# from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

from joblib import dump, load

from smogn import smoter
from imbalance_metrics import regression_metrics as rm

import seaborn as sns

# These packages are used when the code to plot the decision tree is uncommented
# See **TREE PLOTTING** section and un-uncomment the code there
from sklearn.tree import plot_tree
from sklearn.tree import export_text

#------------------------------------------------------------------------------
# Define a function that calcualte error metrics from predicted and actual values
def reg_model_metrics(actual,pred):
    MSE = mean_squared_error(actual,pred)
    RMSE = np.sqrt(MSE)
    actual_mean = np.mean(actual)
    RRMSE = 100*RMSE/actual_mean
    MAE = mean_absolute_error(actual, pred)
    R2 = r2_score(actual,pred)
    D2 = d2_absolute_error_score(actual, pred)
    MAXErr = max_error(actual, pred)
    EVS = explained_variance_score(actual, pred)
    return MSE, RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS    

# Define a function that calculates weighted error metrics from "imbalance_metrics" package
# Package can be found at: https://github.com/paobranco/ImbalanceMetrics
def weighted_reg_model_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef, rel_matrix):
    actual = actual.to_numpy()
    # pred = pred.to_numpy()
    
    actual = actual.flatten()
    # pred = pred.flatten()
    
    wmse = rm.phi_weighted_mse(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    wmae = rm.phi_weighted_mae(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    wr2 = rm.phi_weighted_r2(actual , pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    wrmse = rm.phi_weighted_root_mse(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    threshold = .05
    ser_t_5 = rm.ser_t(actual, pred, threshold, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    threshold = .7
    ser_t_70 = rm.ser_t(actual, pred,threshold, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    sera = rm.sera(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
    return wmse, wmae, wrmse, wr2, ser_t_5, ser_t_70, sera

# Function to calculate and organize metrics
def calculate_and_organize_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef, rel_matrix, smogn_flag):
     
    # Calculate standard error metrics
    mse, rmse, rrmse, mae, r2, d2, max_err, evs = reg_model_metrics(actual, pred)
    
    # Calculate weighted error metrics
    WMSE, WMAE, WRMSE, WR2, SER_t_5, SER_t_70, SERA = weighted_reg_model_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef, rel_matrix)
    
    # Organize metrics into a dictionary
    metrics_dict = {
        "MSE": mse,
        "RMSE": rmse,
        "RRMSE": rrmse,
        "MAE": mae,
        "R2": r2,
        "D2": d2,
        "MaxErr": max_err,
        "EVS": evs,
        "WMSE": WMSE,
        "WMAE": WMAE,
        "WRMSE": WRMSE,
        "WR2": WR2,
        "SER_t_5": SER_t_5,
        "SER_t_70": SER_t_70,
        "SERA": SERA,
    }
   
    return metrics_dict

# Display prediction resutls using 1:1 scatter plot
def scatter_plot(actual, pred, title, var_of_int, time, label, m, sl, stri, rel_method, rel_xtrm_type, rel_coef, rel_matrix, smogn_flag):
    # Color Matching for clarity
    if var_of_int == "Methane (ppm)":
        color = 'blue'
        cm = 'Blues'
    elif var_of_int == "DMS (MS)":
        color = 'purple'
        cm = 'Purples'
    elif var_of_int == "Benzene (E_MS)":
        color = 'saddlebrown'
        cm = 'YlOrBr'
    elif var_of_int == 'Toluene (B)':
        color = 'red'
        cm = 'Reds'
    elif var_of_int == 'Isoprene (E_B)':
        color = 'mediumseagreen'
        cm = 'Greens'
    else:
        color = 'orange'
        cm = 'Oranges'    
    
    if not smogn_flag:
        MSE, RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS = reg_model_metrics(actual, pred)
            
        fig,ax = plt.subplots(figsize=(8, 6))
        ax.scatter(actual, pred, edgecolors=(0,0,0), c=color)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        text = r"R2 = %.2f" % (R2); text += "\n";
        # text += r"D2 = %.2f" % (D2); text += "\n";
        text += r"MAE = %.2f" % (MAE); text += "\n";
        text += r"MSE = %.2f" % (MSE); text += "\n";
        text += r"RMSE = %.2f" % (RMSE);     
        plt.annotate(text, xy=(0.01, 0.85), xycoords='axes fraction',color='black', fontsize=10,bbox=dict(facecolor='none', edgecolor='none'))
        ax.set_xlabel('Measured ' + str.split(var_of_int)[0] + time + " " + label)
        ax.set_ylabel('Predicted ' + str.split(var_of_int)[0] + time + " " + label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + m + "/Scatter_" + m + sl + str.split(var_of_int)[0] + time + stri + ".png")
        plt.show()
    else:
        WMSE, WMAE, WRMSE, WR2, SER_t_5, SER_t_70, SERA = weighted_reg_model_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef, rel_matrix)
        
        if stri == "_Train":
            # print('No Relevence Function')
            numpy_y_actual = actual.to_numpy()
            flattened_numpy_y_actual = numpy_y_actual.flatten()
            
            samp_weights_actual = rm.calculate_phi(flattened_numpy_y_actual, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef, ctrl_pts=rel_matrix)
            
            color = samp_weights_actual
        
        fig,ax = plt.subplots(figsize=(8, 6))
        ax.scatter(actual, pred, edgecolors=(0,0,0), c=color, cmap=cm)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        text = r"W_R2 = %.2f" % (WR2); text += "\n"; text += r"W_MAE = %.2f" % (WMAE); text += "\n"; text += r"W_MSE = %.2f" % (WMSE); text += "\n"; text += r"W_RMSE = %.2f" % (WRMSE);      
        plt.annotate(text, xy=(0.05, 0.85), xycoords='axes fraction',color='black', fontsize=10,
                     bbox=dict(facecolor='none', edgecolor='none'))
        ax.set_xlabel('Measured ' + str.split(var_of_int)[0] + time + " " + label)
        ax.set_ylabel('Predicted ' + str.split(var_of_int)[0] + time + " " + label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + m + "/Scatter_" + m + sl + str.split(var_of_int)[0] + time + stri + ".png")
        plt.show()
        
# Permutation importance calclation and plot
def plot_permutation_importance_scores(input_model, input_x, input_y, m, sl, title):
    xl = input_x.columns # input_x labels
    # yl = str.split(input_y.columns[0])[0] # #this line for not smogn
    yl = input_y.name # This line for smogn
    
    # Calculate the Variable Importance
    perm_imp = permutation_importance(input_model, input_x, input_y, n_repeats=10, random_state=rs, n_jobs=-1)
    # Sort inices
    sorted_idx = perm_imp.importances_mean.argsort()

    # Plot a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(perm_imp.importances[sorted_idx].T,
               vert=False,
               labels=xl[sorted_idx])
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + m + "/Importances_" + m + sl + yl + "_" + t + ".png")
    plt.show()
    plt.clf()
    return perm_imp, sorted_idx

# Function to make manual phi relavance values for SMOGN (EXPERIEMENTAL!)
# This function should undersample samples in the background concentration of a VOC
# and oversample example points at a specified range and step above background concentration
def create_rg_mtrx(umin, umax, u_spacing, omin, omax, o_spacing):
    # Generate undersampled values
    under_sampled = np.arange(umin, umax + u_spacing, u_spacing)
    # Generate oversampled values
    over_sampled = np.arange(omin, omax + o_spacing, o_spacing)
    # Create the rg_mtrx
    rg_mtrx = [[value, 1, 0] for value in over_sampled] + [[value, 0, 0] for value in under_sampled]
    return rg_mtrx

# Weighted loss functions
# def custom_squared_error(y_true, y_pred, sample_weights):
#     squared_errors = (y_true - y_pred) ** 2
#     weighted_errors = squared_errors * sample_weights
#     return np.mean(weighted_errors)
def custom_absolute_error(y_true, y_pred, sample_weights):
    absolute_errors = np.abs(y_true - y_pred)
    weighted_errors = absolute_errors * sample_weights
    return np.mean(weighted_errors)

#-----------------------------------------------------
# VOC variable for prediction
v0 = "DMS (MS)"  # 3 pptv
v1 = "Methane (ppm)" # 1.920 ppmv
v2 = "Benzene (E_MS)" 
v3 = "Toluene (B)" 
v4 = "Isoprene (E_B)" # 0.1
v5 = "Ethane (E)"
v6 = "H-1211 (C_D)" # in testing
v7 = "H-1301 (C)"  # in testing
v8 = "CH3Br (C_MS)" # in testing

# Time variable for prediction 
t0 = "T1"
t1 = "T-24"

# # Model VOCs
vocs = ["DMS (MS)", "Methane (ppm)", "Benzene (E_MS)", "Toluene (B)", "Isoprene (E_B)", "Ethane (E)"]

# Model VOCs
vocs = ["Isoprene (E_B)"]

# Model Time variables (Must be "T0" and/or "T-24")
times = ["T0","T-24"]

models = ['RFR'] # ['RFR', 'GB'] #(EXPERIMENTAL!) 


#-----------------------------------Relevance Functions------------------------
"""
Weighting the phi relavence should is recommended to done by a domain expert on a per VOC basis. Since I am no domain expert on atmospheric chemistry, these relevenece and over/undersampling boundaries are based on what my first interpretation of these relevence boundaries should be! 

Additionaly there are are matricies that have the suffix '_SMOGN' that are difference from the relevance matrics used for weighted metrics, the suffixed matricies were hand tuned and the 'auto' feature within the SMOGN package look to me to oversample at an unresonably high boundary. (as we still want to represent the lower ranges of whole air sample correctly ie. most canisters that are opened will still report near background/Limit of detection concentration)

There is also a DISCLAIMER on this section. While the PVOCAL project currently used weighted metrics provided by Paula Branco, (https://github.com/paobranco/ImbalanceMetrics) in the future, PVOCAL could certainly make use of Utility Based Regression in its predictions, and although it is not implemented yet, it would certainly provide a good metric for this model
""" 
REL_MATRIX_DMS_SMOGN = [
    [.5,  0, 0],
    # [2,1,0],
    # [3.018804496,  1, 0],
    # [5.3,  1, 0],
    # [6, 1, 0],
    # [9.8, 1, 0],
    # [13.2, 1, 0],
    [23.8,  1, 0],
    # [46.69,  1, 0],
    # [92.47,  1, 0],
    # [150,  1, 0],
    ]

REL_MATRIX_CH4_SMOGN = [
    [1.763, 0 ,0],
    # [2.356, 1 ,0],
    [2.480, 1 ,0],
    # [3.231, 1 ,0],
    # [3.356, 1 ,0],
    # [3.481, 1 ,0],
    # [3.5, 1 ,0]
    ]

REL_MATRIX_BENZENE_SMOGN = [
    [3.3,0,0],
    [24,0,0],
    # [23,.5,0],
    # [38.51,.5,0],
    # [43,1,0],
    [100.278872,1,0],
    # [157.07,1,0],
    # [214.11,1,0],
    # [271.14,1,0],
    # [328.179,1,0],
    # [385.215,1,0],
    # [442.25,1,0],
    # [499.19,1,0],
    # [556.32,1,0],
    # [613.36,1,0],
    [670,1,0],
    # [727,1,0],
    # [784,1,0],
    # [841,1,0],
    # [898,1,0],
    # [955,1,0],
    # [1012,1,0],
    # [1069,1,0],
    # [1126,1,0],
    # [1183,1,0],
    # [1240,1,0],
    # [1297,1,0],
    # [1354,1,0],
    ]

REL_MATRIX_TOLUENE_SMOGN = [
    [3,  0, 0],
    # [11, 0, 0],    
    # [22, .5, 0],
    # [55.46,.5,0],
    [107.49, 1, 0],
    # [155,  1, 0],
    # [254, 1, 0],
    # [353, 1, 0],
    # [452, 1, 0],
    # [551, 1, 0],
    # [650, 1, 0],
    # [749, 1, 0],
    # [848, 1, 0],
    # [947, 1, 0]
    ]

REL_MATRIX_ISOPRENE_SMOGN = [
    [3,0,0],
    # [6.84,.5,0],
    # [16,.5,0],
    # [45.82,.5,0],
    # [98,1,0],
    [146,1,0],
    # [140.07,1,0],
    # [234.11,1,0],
    # [422.25,1,0],
    # [516.02,1,0],
    # [610.36,1,0],
    # [704,1,0],
    # [798,1,0],
    # [892,1,0],
    # [986,1,0],
    # [1080,1,0],
    # [1268,1,0],
    # [1362,1,0]
    ] 

REL_MATRIX_ETHANE_SMOGN = [
    [247,0,0],
    # [675,0.5,0],
    # [920,.5,0],
    # [1254,.5,0],
    # [1427,1,0],
    [3090,1,0],
    # [2763,1,0],
    # [4754,1,0],
    # [5863,1,0],
    # [6972,1,0],
    # [8081,1,0],
    # [9190,1,0],
    # [10299,1,0],
    # [11408,1,0],
    # [12517,1,0],
    # [13626,1,0]
    ]

REL_MATRIX_DMS = [
    [0.5,  .6, 0],
    [2,  .8, 0],
    [3,  1, 0],
    [4,  1, 0],
    [9.8, 1, 0],
    [19.35, 1,0],
    [23.8,  1, 0],
    [46.69,  1, 0],
    [69.58,  1, 0],
    [92.47,  1, 0],
    [115.36,  1, 0],
    [138.25,  1, 0],
    [161.14,  1, 0],
    [184.03,  1, 0],
    ]

REL_MATRIX_CH4 = [
    [1.75, .6 ,0],
    [1.857, .8 ,0],
    [1.923, 1 ,0],
    [1.939, 1 ,0],
    [2.981, 1 ,0],
    [3.106, 1 ,0],
    [3.231, 1 ,0],
    [3.356, 1 ,0],
    [3.481, 1 ,0],
    [3.5, 1 ,0]
    ]

REL_MATRIX_BENZENE = [
    [3,.6,0],
    [13,.8,0],
    [23,1,0],
    [38.7,1,0],
    [43,1,0],
    [100,1,0],
    [157.07,1,0],
    [214.11,1,0],
    [271.14,1,0],
    [328.179,1,0],
    [385.215,1,0],
    [442.25,1,0],
    [499.19,1,0],
    [556.32,1,0],
    [613.36,1,0],
    [670,1,0],
    [727,1,0],
    [784,1,0],
    [841,1,0],
    [898,1,0],
    [955,1,0],
    [1012,1,0],
    [1069,1,0],
    [1126,1,0],
    [1183,1,0],
    [1240,1,0],
    [1297,1,0],
    [1354,1,0],
    ]

REL_MATRIX_TOLUENE = [
    [3,  .6, 0],
    [11, .8, 0],    
    [23, 1, 0],
    [56.46,1,0],
    [57.4, 1, 0],
    [155,  1, 0],
    [254, 1, 0],
    [353, 1, 0],
    [452, 1, 0],
    [551, 1, 0],
    [650, 1, 0],
    [749, 1, 0],
    [848, 1, 0],
    [947, 1, 0]
    ]

REL_MATRIX_ISOPRENE = [
    [3,.6,0],
    [7,.8,0],
    [18,1,0],
    [47.6,1,0],
    [50,1,0],
    [140.07,1,0],
    [234.11,1,0],
    [422.25,1,0],
    [516.02,1,0],
    [610.36,1,0],
    [704,1,0],
    [798,1,0],
    [892,1,0],
    [986,1,0],
    [1080,1,0],
    [1268,1,0],
    [1362,1,0]
    ] 

REL_MATRIX_ETHANE = [
    [247,0.6,0],
    [675,0.8,0],
    [920,1,0],
    [1254,1,0],
    [1427,1,0],
    [2536,1,0],
    [3645,1,0],
    [4754,1,0],
    [5863,1,0],
    [6972,1,0],
    [8081,1,0],
    [9190,1,0],
    [10299,1,0],
    [11408,1,0],
    [12517,1,0],
    [13626,1,0]
    ]
#------------------------------------------------------------------------------

      

#------------------------------Model Parameters--------------------------------
# # Weights for weighted Gradient Boosted loss function
# ## both 1 means all values are of equal weight loss function
# minority_weight = 1  # Custom weight for minority cases 
# majority_weight = 1  # Custom weight for majority cases
# perc_thres = .75 # threshold for weighted loss function

rel_func = True
PDP = True
smog = True # perform smogn algorithm?
tree_plotting = False # perform simple tree plot for diagnosic vizualization?
# phi relevence function arguments # FOR BOTH SMOGN AND WEIGHTED METRICS 
REL_THRES = .30 
REL_METHOD = 'manual' 
REL_XTRM_TYPE = 'high' 
REL_COEF = 3

# Specify Random state?
rs = 2 

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# Read in CSV file with all of the data we need. Meteorology variables + Pathdata + VOC data
file_path = r"C:\Users\vwgei\Documents\PVOCAL\data\PVOCAL_data_wtime.csv"

# Load data from CSV into a pandas DataFrame.
data = pd.read_csv(file_path)

# SMOGN vs Not SMOGN formatting for output plots 
if smog:
    smog_label = "_smogn_"
else:
    smog_label = "_"

# Main PVOCAL loop
for model in models:
    for voc in vocs:
        for time in times:
            var_of_int = voc
            t = time
            
            # Match unit labels to VOC
            if var_of_int == "Methane (ppm)":
                label = "ppmv"
            else:
                label = "pptv"
            
            if var_of_int == "Methane (ppm)":
                MODEL_REL_MATRIX = REL_MATRIX_CH4    
                MODEL_REL_MATRIX_SMOGN = REL_MATRIX_CH4_SMOGN
            elif var_of_int == "DMS (MS)":
                MODEL_REL_MATRIX = REL_MATRIX_DMS    
                MODEL_REL_MATRIX_SMOGN = REL_MATRIX_DMS_SMOGN
            elif var_of_int == "Benzene (E_MS)":
                MODEL_REL_MATRIX = REL_MATRIX_BENZENE  
                MODEL_REL_MATRIX_SMOGN = REL_MATRIX_BENZENE_SMOGN
            elif var_of_int == 'Toluene (B)':
                MODEL_REL_MATRIX = REL_MATRIX_TOLUENE  
                MODEL_REL_MATRIX_SMOGN = REL_MATRIX_TOLUENE_SMOGN
            elif var_of_int == 'Isoprene (E_B)':
                MODEL_REL_MATRIX = REL_MATRIX_ISOPRENE   
                MODEL_REL_MATRIX_SMOGN = REL_MATRIX_ISOPRENE_SMOGN
            else:
                MODEL_REL_MATRIX = REL_MATRIX_ETHANE
                MODEL_REL_MATRIX_SMOGN = REL_MATRIX_ETHANE_SMOGN
                
            # Gets rid of bad values in the predicted variable
            data = data.dropna(subset = var_of_int) 
            
            # Set input variables for model T0 endpoints vs T-24 endpoints
            if (t == "T0"):
                # Initial point data at time: 0 hr
                predictionBaseVars = data.iloc[:, 3:14] 
            else:
                # Backwards trajctory point data at time: -24hr # with distance features
                predictionBaseVars = data.iloc[:, 17:30]
    
            # Get the VOC of interest
            prediction_var = data.loc[:, [var_of_int]]
            
            # """
            # -Create a binary indicator for minority cases (e.g., values above a certain threshold)
            # This is a manual version of sample weights not from imbalenced metrics package!!
            # Controlled through model parameters - Based on a discrete threshold
            # """
            # threshold = np.percentile(prediction_var, perc_thres)
            # is_minority = prediction_var > threshold
            # is_majority = prediction_var < threshold
            
            # # Assign higher weights to minority cases
            # sample_weights = np.ones_like(prediction_var)
            # sample_weights[is_minority] = minority_weight  # Assign a higher weight to minority cases
            # sample_weights[is_majority] = majority_weight # Assign a lower weight to majority cases
            
            # Split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(predictionBaseVars, prediction_var, test_size=0.2, random_state=rs) 
            

          

            if smog:
                # SMOGN algorithm for unbalenced regression
                # Example here: https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb
                
                # temporarily join x_train and y_train for SMOGN algorithm
                dftemp = x_train.merge(y_train, how='inner', left_index=True, right_index=True)
                               

                PVOCAL_smognf = smoter(
                    ## main arguments
                    data = dftemp.reset_index(drop=True),        ## pandas dataframe
                    y = var_of_int,          ## string ('header name')
                    k = 5,                    ## positive integer (k < n)
                    pert = 0.04,              ## real number (0 < R < 1)
                    samp_method = 'extreme',  ## string ('balance' or 'extreme')
                    under_samp = True,        ## Undersample majority class (keep this True)
                    drop_na_col = False,       ## boolean (True or False)
                    drop_na_row = True,       ## boolean (True or False)
                    replace = False,          ## boolean (True or False)
                    
                    ## phi relevance arguments
                    rel_thres = REL_THRES,         ## real number (0 < R < 1)
                    rel_method = REL_METHOD,    ## string ('auto' or 'manual')
                    rel_xtrm_type = REL_XTRM_TYPE, ## high extremes are the minority
                    rel_coef = REL_COEF,        ## increase or decrease box plot extremes
                    rel_ctrl_pts_rg = MODEL_REL_MATRIX_SMOGN ## 2d array (format: [x, y])
                )
                
                PVOCAL_smognf = PVOCAL_smognf.dropna()
                
                PVOCAL_smogn = PVOCAL_smognf[PVOCAL_smognf[var_of_int] > 0.000001] # get rid of some invalid samples being generated with very low x^-287 floating point values
                
                
              
                # Format legend
                histplotlabel1 = var_of_int.split()[0] + t + ' before SMOGN algorithm'
                histplotlabel2 = var_of_int.split()[0] + t + ' after SMOGN algorithm'
                
                # Histogram distribution plots
                sns.histplot(prediction_var, kde=True, label=histplotlabel1, color='blue')
                sns.histplot(PVOCAL_smogn[var_of_int], kde=True, label=histplotlabel2, color='orange')
                
                plt.legend()
                plt.title('Distribution of ' + var_of_int.split()[0] + t + "Training Set")
                plt.xlabel(var_of_int.split()[0] + t + " " + label)
                plt.ylabel('Sample Count')
                
                # Save the distribution plot
                plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + model + "/DistPlot_" + model + smog_label + str.split(var_of_int)[0] + "_" + t + ".png")
                plt.show()
                plt.clf()
                
                
                # # save the correlation matrix plot
                # plt.figure(figsize=(20,18))
                # cor=PVOCAL_smogn.corr(method='kendall')
                # cor = abs(cor)
                # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
                # plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + model + "/CorrMatrix_" + model + smog_label + str.split(var_of_int)[0] + "_" + t +".png")
                # plt.show()
                
                # Redefine/re-separate x_train and y_train
                x_train = PVOCAL_smogn.iloc[:, 0:(len(PVOCAL_smogn.columns)-1)]
                y_train = PVOCAL_smogn.iloc[:,-1]    
                
            #-------------------------END SMOGN--------------------------------
                     
            numpy_y_train = y_train.to_numpy()
            flattened_numpy_y_train = numpy_y_train.flatten()

            samp_weights_train = rm.calculate_phi(flattened_numpy_y_train, method = REL_METHOD, ctrl_pts=MODEL_REL_MATRIX) #xtrm_type = REL_XTRM_TYPE, coef = REL_COEF,  #xtrm_type = REL_XTRM_TYPE, coef = REL_COEF)

            fig,ax = plt.subplots()
            ax.scatter(y_train,x_train.iloc[:,9], c=samp_weights_train, cmap='Reds')
            ax.set_xlabel(str.split(var_of_int)[0] + label)
            ax.set_ylabel('Atltitude in meters')
            ax.set_xscale('log')
            # ax.set_yscale('log')
            plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/Relavence_func_" + t + str.split(var_of_int)[0] + "_" + t + ".png")
            ax.set_title('Relevence Function visualization')
            plt.show()              

            
            if model == 'RFR':
                # # Color matching to different VOCs for clarity    
                # if var_of_int == "Methane (ppm)":
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.9, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="friedman_mse", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                # elif var_of_int == "DMS (MS)":
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.2, max_depth=11, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.9, max_depth=11, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                # elif var_of_int == "Benzene (E_MS)":
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.6, max_depth=11, min_samples_split=15, min_samples_leaf=5, criterion="absolute_error", min_weight_fraction_leaf=0.0002, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.9, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="friedman_mse", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                #     #     PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.3, max_depth=11, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.008, bootstrap=True, oob_score=True, random_state=rs)
                #     # else:
                #     #     PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.3, max_depth=11, min_samples_split=2, min_samples_leaf=1, criterion="friedman_mse", min_weight_fraction_leaf=0.008, bootstrap=True, oob_score=True, random_state=rs)
                
                # elif var_of_int == 'Toluene (B)':
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.9, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.0, bootstrap=True, oob_score=True, random_state=rs)
                
                # elif var_of_int == 'Isoprene (E_B)':
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.3, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.0, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.4, max_depth=6, min_samples_split=25, min_samples_leaf=13, criterion="absolute_error", min_weight_fraction_leaf=0.002, bootstrap=True, oob_score=True, random_state=rs)
                # elif var_of_int == 'Ethane (E)':
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.9, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.014, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.9, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.013, bootstrap=True, oob_score=True, random_state=rs)
                # else:                    
                PVOCAL_est = RandomForestRegressor(random_state=rs)
                                
                param_grid = {'n_estimators': [1000],
                               'max_features': [.25],
                               'max_depth': [None],#7
                               'min_samples_split': [2], #3
                               'min_samples_leaf': [1], #2
                               'criterion': ['absolute_error'],
                               'bootstrap': [True],
                               'oob_score': [True],
                               'min_weight_fraction_leaf': [0.01],
                               'ccp_alpha': [0.00]
                               }
                
            elif model == 'GB':
                # if var_of_int == 'DMS (MS)':
                #     if time == "T0":
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=550, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                #     else:
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=365, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                # elif var_of_int == 'Methane (ppm)':
                #     if time == "T0":
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=700, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                #     else:
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=500, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                # elif var_of_int == 'Benzene (E_MS)':
                #     if time == 'T0':
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=1800, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.2, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                #     else: 
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=600, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.2, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                        
                # elif var_of_int == 'Toluene (B)':
                #     if time == 'T0':
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=1100, max_features=0.5, min_weight_fraction_leaf=0.05, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                #     else: 
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=1700, max_features=0.8, min_weight_fraction_leaf=0.05, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                        
                # elif var_of_int == 'Isoprene (E_B)':
                #     if time == 'T0':
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=500, max_features=0.5, min_weight_fraction_leaf=0.00, subsample=0.3, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                #     else: 
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=200, max_features=0.8, min_weight_fraction_leaf=0.00, subsample=0.8, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                        
                        
                # elif var_of_int == 'Ethane (E)':
                #     if time == 'T0':
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=800, max_features=0.3, min_weight_fraction_leaf=0.01, subsample=1, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                #     else: 
                #         PVOCAL_EST = GradientBoostingRegressor(loss='huber', n_estimators=800, max_features=0.5, min_weight_fraction_leaf=0.01, subsample=0.8, learning_rate=0.01, max_depth=3, alpha=0.9, random_state=rs)
                # else:
                PVOCAL_est = GradientBoostingRegressor(random_state=rs)    
                    
                param_grid = {
                    'loss': ['huber'],
                    # 'warm_start':[True],
                    # 'n_iter_no_change': [50],
                    # 'tol':[0.01],
                    'max_features': [.3,.5,.8, 1],
                    # 'validation_fraction': [.1],
                    'min_weight_fraction_leaf': [0.01,0.005],
                    'subsample': [0.3,0.8,1.0],
                    'n_estimators': [1000],
                    'learning_rate': [0.01],
                    'max_depth': [3],
                    'alpha' : [0.9]
                }

                ## ADA Boost implementation
            
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=PVOCAL_est, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=12)
            
            fit_params = {'sample_weight': samp_weights_train}
            
            # Fit the model to the training data
            grid_search.fit(x_train, y_train.values.ravel(),**fit_params)
            
            print(model)
            print(var_of_int)
            print(t)
            bestParam = grid_search.best_params_
            print(bestParam)
            
            PVOCAL_EST = grid_search.best_estimator_
            
            tempcsv = pd.DataFrame(grid_search.cv_results_)
            tempcsv.to_csv("C:/Users/vwgei/Documents/PVOCAL/data/gridSearchCV_Results/GridSearchResults" + model + smog_label + var_of_int.split()[0] + t + ".csv", index=False)
            # PVOCAL_EST.fit(x_train, y_train.values.ravel(), samp_weights_train)
            
            dump(PVOCAL_EST, "C:/Users/vwgei/Documents/PVOCAL/bestmodels/" + model + smog_label + var_of_int.split()[0] + t + ".joblib")
            
            # PVOCAL_EST.fit(x_train,y_train.values.ravel())
                      
            # Make predictions on the training data
            y_pred_train = PVOCAL_EST.predict(x_train)
            # Make predictions on the testing data.
            y_pred_test = PVOCAL_EST.predict(x_test)
            
            # if model == 'RFR':
            #     path = PVOCAL_EST.cost_complexity_pruning_path(x_train, y_train)
            #     ccp_alphas, impurities = path.ccp_alphas, path.impurities
                
            #     rfrs = []
            #     for ccp_alpha in ccp_alphas:
            #         rfr = RandomForestRegressor(**bestParam, ccp_alpha=ccp_alpha)
            #         rfr.fit(x_train, y_train)
            #         rfrs.append(rfr)
                
            #     train_scores = [rfr.score(x_train, y_train) for rfr in rfrs]
            #     test_scores = [rfr.score(x_test, y_test) for rfr in rfrs]
                
            #     fig, ax = plt.subplots()
            #     ax.set_xlabel("alpha")
            #     ax.set_ylabel("accuracy")
            #     ax.set_title("Accuracy vs alpha for training and testing sets")
            #     ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            #     ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            #     ax.legend()
            #     plt.show()
            
            if model == 'GB':
                # Compute the cumulative MSE on the test set at each stage
                test_errors = [mean_absolute_error(y_test, y_pred) for y_pred in PVOCAL_EST.staged_predict(x_test)]
                train_errors = [mean_absolute_error(y_train, y_pred) for y_pred in PVOCAL_EST.staged_predict(x_train)]
                
                # Plot the errors over boosting stages
                plt.plot(np.arange(1, len(test_errors) + 1), test_errors, label='Test Set MAE', color='orange')
                plt.plot(np.arange(1, len(train_errors) + 1), train_errors, label='Training Set MAE', color='blue')
                plt.xlabel('Boosting Iterations')
                plt.ylabel('Mean absolute Error')
                plt.title('Train and Test Mean Absolute Error: ' + var_of_int.split()[0] + t)
                plt.legend()
                plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + model + "/LC_plot_" + model + smog_label + str.split(var_of_int)[0] + "_" + t +".png")
                plt.show()
                
                # Calculate the R^2 over boosting stages
                train_r2_values = []
                test_r2_values = []
                # Fit the model to the training data and compute R-squared values at each stage
                for i, y_pred_tr in enumerate(PVOCAL_EST.staged_predict(x_train)):
                    r2_train = r2_score(y_train, y_pred_tr)
                    train_r2_values.append(r2_train)
                
                for i, y_pred_te in enumerate(PVOCAL_EST.staged_predict(x_test)):
                    r2_test = r2_score(y_test, y_pred_te)
                    test_r2_values.append(r2_test)
                
                # Plot the R-squared values over boosting stages
                plt.plot(np.arange(1, len(train_r2_values) + 1), train_r2_values, label='Training R-squared', color='blue')
                plt.plot(np.arange(1, len(test_r2_values) + 1), test_r2_values, label='Testing R-squared', color='green')
                plt.xlabel('Boosting Iterations')
                plt.ylabel('R-squared')
                plt.title('Train and Test R-squared: ' + var_of_int.split()[0] + t)
                plt.legend()
                plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + model + "/R2_plot_" + model + smog_label + str.split(var_of_int)[0] + "_" + t +".png")
                plt.show()
            

            
            #---------------------------Model Metrics--------------------------
            
            """
            ##
            ### Calculate SciKitLearn model scoring metrics.
            ##

            MSE, RMSE, RRMSE : Average squared residual size for the model
            R2 : How good of an overall fit is the model to the data, the coefficient of determination
            D2 : Generalization of r2 (skill score) squared error replaced by deviance mean absolute error
            Max Error 'MaxErr' : Largest Error between prediction and observed value
            Explained Varience Score 'EVS' : Explained variance score does not account for systematic offset in the prediction, Predictor is not biased (not bias if EXV=R2)

            """

            # Calculate and organize metrics for training set
            train_metrics = calculate_and_organize_metrics(y_train, y_pred_train, REL_METHOD, REL_XTRM_TYPE, REL_COEF, MODEL_REL_MATRIX, smog)
            
            # Calculate and organize metrics for testing set
            test_metrics = calculate_and_organize_metrics(y_test, y_pred_test, REL_METHOD, REL_XTRM_TYPE, REL_COEF, MODEL_REL_MATRIX, smog)
            
            # Create DataFrames for training and testing metrics
            train_metrics_df = pd.DataFrame(train_metrics, index=["Training"])
            test_metrics_df = pd.DataFrame(test_metrics, index=["Testing"])
            
            # Concatenate DataFrames if needed
            all_metrics_df = pd.concat([train_metrics_df, test_metrics_df])
            
            
            # Round the values in the DataFrames to 3 decimals
            rounded_train_df = train_metrics_df.round(3)
            rounded_test_df = test_metrics_df.round(3)
            
            print("Training metrics     |     Testing Metrics")
            # Find the maximum length of the column names for even spacing
            max_len_train = max(len(col) for col in rounded_train_df.columns)
            max_len_test = max(len(col) for col in rounded_test_df.columns)
            
            # Print the formatted output for train and test metrics side by side
            for train_column, test_column in zip(rounded_train_df.columns, rounded_test_df.columns):
                train_value = rounded_train_df[train_column].values[0]
                test_value = rounded_test_df[test_column].values[0]
            
                # Calculate the number of spaces needed for even spacing
                train_spaces = " " * (max_len_train - len(train_column))
                test_spaces = " " * (max_len_test - len(test_column))
            
                print(f"{train_column}:{train_spaces}{train_value}\t|\t{test_column}:{test_spaces}{test_value}")
            
            # If the total number of columns is odd, print a newline to avoid ending with a tab
            if len(rounded_train_df.columns) % 2 != 0:
                print()
                
            if model == 'RFR':
                print('Random Forest Out of Bag Score: ')
                print(round(PVOCAL_EST.oob_score_,3))
            
            
            # Export to CSV
            all_metrics_df.to_csv("C:/Users/vwgei/Documents/PVOCAL/data/ErrorMetrics/ErrMetrics_" + model + var_of_int + smog_label + time + ".csv")

            # Visulize predictions on training set 
            title = 'Training Results: ' + str.split(var_of_int)[0] + " " +  time 
            scatter_plot(y_train, y_pred_train, title, var_of_int, t, label, model, smog_label, "_Train", REL_METHOD, REL_XTRM_TYPE, REL_COEF, MODEL_REL_MATRIX, smog)

            # Visulize predictions on testing set
            title = 'Testing Results: ' + str.split(var_of_int)[0] + " " + time
            scatter_plot(y_test, y_pred_test, title, var_of_int, t, label, model, smog_label, "_Test", REL_METHOD, REL_XTRM_TYPE, REL_COEF, MODEL_REL_MATRIX, smog)
            
            if tree_plotting:
                # Visualize decision tree - Increases run time significantly
                fig = plt.figure(figsize=(30, 20))
                plot_tree(PVOCAL_EST.estimators_[0], 
                          filled=True, impurity=True, 
                          rounded=True)
                fig.savefig(r"C:\Users\vwgei\Documents\PVOCAL\plots\rfrtree.png")
            
            # Generate permutation importances using function
            perm_importances, perm_importances_index = plot_permutation_importance_scores(PVOCAL_EST, x_train, y_train, model, smog_label, title="Permutation Importances using Train: " + var_of_int.split()[0])
            

#-----------------------------------Partial Dependence-------------------------            
            print("perm impt indx = ")
            print(perm_importances_index)
            # Get feature importances
            feature_importances = PVOCAL_EST.feature_importances_
            
            # Get indices of the two most important features
            top_two_indices = np.argsort(feature_importances)[-2:]
            top_two_permutation_indeces = perm_importances_index[-2:]
            # print()
            
            # Print the indices and names of the two most important features
            print("Feature names:", x_train.columns[top_two_indices])
                        
            
            
            if PDP:
                plt.clf()    
                #From SKlearn PDP tutorial!   
                top_two_feat = x_train.columns[top_two_permutation_indeces] 
                
                features_info_top_two = {
                    "features": [top_two_feat[0], top_two_feat[1],(top_two_feat[0], top_two_feat[1])],
                    "kind" : "average",
                }
                _, ax = plt.subplots(ncols=3, figsize=(10, 4), constrained_layout=True)
                            
                # common_params = {
                #     "sample_weight":samp_weights_train
                #     }
                
                display = PartialDependenceDisplay.from_estimator(
                PVOCAL_EST,
                x_train,
                **features_info_top_two,
                ax=ax,
                # **common_params,
                )
                    
                _ = display.figure_.suptitle(("1-way vs 2-way Partial Dependance for \n" + var_of_int.split()[0] + " " + time + " of " + str(top_two_feat[0]) + " and " + str(top_two_feat[1])),fontsize=16)
                plt.show()


        