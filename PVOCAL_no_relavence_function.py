# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:12:48 2023



@author: Victor Geiser
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
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
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
def weighted_reg_model_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef):
    actual = actual.to_numpy()
    # pred = pred.to_numpy()
    
    actual = actual.flatten()
    # pred = pred.flatten()
    
    wmse = rm.phi_weighted_mse(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    wmae = rm.phi_weighted_mae(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    wr2 = rm.phi_weighted_r2(actual , pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    wrmse = rm.phi_weighted_root_mse(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    threshold = .05
    ser_t_5 = rm.ser_t(actual, pred, threshold, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    threshold = .7
    ser_t_70 = rm.ser_t(actual, pred,threshold, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    sera = rm.sera(actual, pred, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
    return wmse, wmae, wrmse, wr2, ser_t_5, ser_t_70, sera

# Function to calculate and organize metrics
def calculate_and_organize_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef, smogn_flag):
    # if smogn_flag:    
        # Calculate standard error metrics
    mse, rmse, rrmse, mae, r2, d2, max_err, evs = reg_model_metrics(actual, pred)
    WMSE, WMAE, WRMSE, WR2, SER_t_5, SER_t_70, SERA = weighted_reg_model_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef)
    
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
def scatter_plot(actual, pred, title, var_of_int, time, label, m, sl, stri, rel_method, rel_xtrm_type, rel_coef, smogn_flag):
    
    # Color Matching for clarity
    if var_of_int == "Methane (ppm)":
        color = 'blue'
        cm = 'blues'
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
        
    MSE, RMSE, RRMSE, MAE, R2, D2, MAXErr, EVS = reg_model_metrics(actual, pred)
    # WMSE, WMAE, WRMSE, WR2, SER_t_5, SER_t_70, SERA = weighted_reg_model_metrics(actual, pred, rel_method, rel_xtrm_type, rel_coef)
    
    # if stri == "_Train":
    #     print('No Relevence Function')
    #     numpy_y_actual = actual.to_numpy()
    #     flattened_numpy_y_actual = numpy_y_actual.flatten()
        
    #     samp_weights_actual = rm.calculate_phi(flattened_numpy_y_actual, method = rel_method,  xtrm_type = rel_xtrm_type, coef = rel_coef)
        
    #     color = samp_weights_actual
    
    # fig,ax = plt.subplots(figsize=(8, 6))
    # ax.scatter(actual, pred, edgecolors=(0,0,0), c=color, cmap=cm)
    # ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    # text = r"W_R2 = %.2f" % (WR2); text += "\n"; text += r"W_MAE = %.2f" % (WMAE); text += "\n"; text += r"W_MSE = %.2f" % (WMSE); text += "\n"; text += r"W_RMSE = %.2f" % (WRMSE);      
    # plt.annotate(text, xy=(0.05, 0.85), xycoords='axes fraction',color='black', fontsize=10,
    #              bbox=dict(facecolor='none', edgecolor='none'))
    # ax.set_xlabel('Measured ' + str.split(var_of_int)[0] + time + " " + label)
    # ax.set_ylabel('Predicted ' + str.split(var_of_int)[0] + time + " " + label)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_title(title)
    # plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + m + "/Scatter_" + m + sl + str.split(var_of_int)[0] + time + stri + ".png")
    # plt.show()
        
    
    fig,ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, pred, edgecolors=(0,0,0), c=color)
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    text = r"R2 = %.2f" % (R2); text += "\n"; 
    text += r"D2 = %.2f" % (D2); text += "\n"; 
    text += r"MAE = %.2f" % (MAE); text += "\n";
    text += r"MSE = %.2f" % (MSE); text += "\n"; 
    text += r"RMSE = %.2f" % (RMSE);     
    plt.annotate(text, xy=(0.03, 0.82), xycoords='axes fraction',color='black', fontsize=10,
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
    yl = str.split(input_y.columns[0])[0] # input_y labels
    # yl = input_y.name
    
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
    return perm_imp, sorted_idx

# Function to make manual phi relavance values for SMOGN
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

    
#------------------------------------------------------------------------------
# VOC variable for prediction
v0 = "DMS (MS)" 
v1 = "Methane (ppbv)" 
v2 = "Benzene (E_MS)" 
v3 = "Toluene (B)" 
v4 = "Isoprene (E_B)"
v5 = "Ethane (E)"
v6 = "H-1211 (C_D)" # in testing
v7 = "H-1301 (C)"  # in testing
v8 = "CH3Br (C_MS)" # in testing

# Time variable for prediction 
t0 = "T0"
t1 = "T-24"

tree_plotting = False # perform simple tree plot for diagnosic vizualization?
 
rs = 63 # Specify Random state?
#100 dms t0
#46 dmst24
#ch4t0 70   8t24
#benzenet0 19 30
# ethane 7 11
# isoprene 2  3 11 14
# toluene 42 45 46 48

# vocs = ["DMS (MS)", "Methane (ppm)", "Benzene (E_MS)", "Toluene (B)", "Isoprene (E_B)", "Ethane (E)"]
# Model VOCs

vocs = ["Methane (ppm)"]#, "Isoprene (E_B)", "Ethane (E)"]

# Model Time variables (Must be "T0" and/or "T-24")
times = ["T0"]

models = ['RFR'] #,'ADA' # All available models

# # Model VOCs
# vocs = ["DMS (MS)"]

# # Model Time variables (Must be "T0" and/or "T-24")
# times = ["T0","T-24"]

# models = ['GB','RFR'] 

#------------------------------------------------------------------------------
#------------------------------SMOGN Parameters--------------------------------
# Weights for weighted Gradient Boosted loss function
## both 1 means all values are of equal weight loss function
minority_weight = 1  # Custom weight for minority cases 
majority_weight = 1  # Custom weight for majority cases
perc_thres = .75 # threshold for weighted loss function


# Svoc = False # Just a specific VOC?
# if Svoc == True:
#     SvocI = v0 # Specify VOC

smog = False # perform smogn algorithm?
# phi relevence function arguments # FOR BOTH SMOGN AND WEIGHTED METRICS 
REL_THRES = .50 
REL_METHOD = 'auto' 
REL_XTRM_TYPE = 'high' 
REL_COEF = 3.00

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
                label = "ppm"
            else:
                label = "ppt"
            
            # Color matching to different VOCs for clarity    
            if var_of_int == "Methane (ppm)":
                color = 'blue'
            elif var_of_int == "DMS (MS)":
                color = 'mediumblue'
            elif var_of_int == "Benzene (E_MS)":
                color = 'saddlebrown'
            elif var_of_int == 'Toluene (B)':
                color = 'sandybrown'
            elif var_of_int == 'Isoprene (E_B)':
                color = 'mediumseagreen'
            else:
                color = 'orange' # Ethane
            
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
            
            # samp_weights_test = rm.calculate_phi(y_test, method = REL_METHOD,  xtrm_type = REL_XTRM_TYPE, coef = REL_COEF)

            if smog:
                # SMOGN algorithm for unbalenced regression
                # Example here: https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb
                
                # temporarily join x_train and y_train for SMOGN algorithm
                dftemp = x_train.merge(y_train, how='inner', left_index=True, right_index=True)
                
                
                # #CH4
                # # rg_matrix parameters:
                # umin, umax, omin, omax = low_background_concentration, high_background_concentration, oversample_min, oversample_max
                
                # u_spacing = ((umin + umax) / 10)
                # o_spacing = ((omin + omax) / 100)
                
                # # specify phi relevance values
                # rg_mtrx = create_rg_mtrx(umin, umax, u_spacing, omin, omax, o_spacing)
                # # print(rg_mtrx)
                
                # DMS hyperparameters: 5, 0.04, balance, rel_thres = .20, rel_coef = .60
                # CH4 hyperparameters: 5 0.04, balance, 0.6 manual
                PVOCAL_smognf = smoter(
                    ## main arguments
                    data = dftemp.reset_index(drop=True),        ## pandas dataframe
                    y = var_of_int,          ## string ('header name')
                    k = 5,                    ## positive integer (k < n)
                    pert = 0.04,              ## real number (0 < R < 1)
                    samp_method = 'balance',  ## string ('balance' or 'extreme')
                    under_samp = True,        ## Undersample majority class (keep this True)
                    drop_na_col = False,       ## boolean (True or False)
                    drop_na_row = True,       ## boolean (True or False)
                    replace = False,          ## boolean (True or False)
                    
                    ## phi relevance arguments
                    rel_thres = REL_THRES,         ## real number (0 < R < 1)
                    rel_method = REL_METHOD,    ## string ('auto' or 'manual')
                    rel_xtrm_type = REL_XTRM_TYPE, ## high extremes are the minority
                    rel_coef = REL_COEF,        ## increase or decrease box plot extremes
                    # rel_ctrl_pts_rg = rg_mtrx ## 2d array (format: [x, y])
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
                plt.title('Distribution of ' + var_of_int.split()[0] + t)
                plt.xlabel(var_of_int.split()[0] + t + " " + label)
                plt.ylabel('Frequency')
                
                # Save the distribution plot
                plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + model + "/DistPlot_" + model + smog_label + str.split(var_of_int)[0] + "_" + t + ".png")
                plt.show()
                
                # save the correlation matrix plot
                plt.figure(figsize=(20,18))
                cor=PVOCAL_smogn.corr(method='spearman')
                cor = abs(cor)
                sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
                plt.savefig("C:/Users/vwgei/Documents/PVOCAL/plots/" + model + "/CorrMatrix_" + model + smog_label + str.split(var_of_int)[0] + "_" + t +".png")
                plt.show()
                
                # Redefine/re-separate x_train and y_train
                x_train = PVOCAL_smogn.iloc[:, 0:(len(PVOCAL_smogn.columns)-1)]
                y_train = PVOCAL_smogn.iloc[:,-1]
                
                #-------------------------END SMOGN--------------------------------
            # # Calculates sample weights for the non-smogn model runs using automatic phi relevence
            # # Also Visualizes sample weights with altitude
            # if not smog:
            #     numpy_y_train = y_train.to_numpy()
            #     flattened_numpy_y_train = numpy_y_train.flatten()
                
            #     samp_weights_train = rm.calculate_phi(flattened_numpy_y_train, method = REL_METHOD,  xtrm_type = REL_XTRM_TYPE, coef = REL_COEF)
                
            #     fig,ax = plt.subplots()
            #     ax.scatter(y_train,x_train.iloc[:,9], c=samp_weights_train, cmap='Reds')
            #     ax.set_xlabel('DMS pptv')
            #     ax.set_ylabel('Atltitude in meters')
            #     ax.set_xscale('log')
            #     # ax.set_yscale('log')
            #     ax.set_title('Relevence Function visualization')
            #     plt.show()
            
            # else:
            #     samp_weights_train = np.ones_like(y_train)
            
            if model == 'RFR':          
                # #Color matching to different VOCs for clarity    
                # #DMS
                if var_of_int == "DMS (MS)":
                    if time == 'T0':
                    #     PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.3, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                    # else:
                    #     PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.3, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.02, bootstrap=True, oob_score=True, random_state=rs)
                        PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.3, max_depth=12, min_samples_split=6, min_samples_leaf=3, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                    else:
                        PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.3, max_depth=12, min_samples_split=4, min_samples_leaf=6, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                        
                # #Methane
                # elif var_of_int == "Methane (ppm)":
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.6, max_depth=6, min_samples_split=6, min_samples_leaf=11, criterion="friedman_mse", min_weight_fraction_leaf=0.0035, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.6, max_depth=6, min_samples_split=6, min_samples_leaf=11, criterion="friedman_mse", min_weight_fraction_leaf=0.0035, bootstrap=True, oob_score=True, random_state=rs)        
                
                # # #Benzene        
                # # elif var_of_int == "Benzene (E_MS)":
                # #     if time == 'T0':
                # #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.3, max_depth=9, min_samples_split=5, min_samples_leaf=3, criterion="friedman_mse", min_weight_fraction_leaf=0.00, bootstrap=True, oob_score=True, random_state=rs)
                # #     else:
                # #         PVOCAL_EST = RandomForestRegressor(n_estimators=1000, max_features=0.3, max_depth=9, min_samples_split=5, min_samples_leaf=3, criterion="friedman_mse", min_weight_fraction_leaf=0.00, bootstrap=True, oob_score=True, random_state=rs)
                # # Benzene        
                # elif var_of_int == "Benzene (E_MS)":
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.3, max_depth=10, min_samples_split=8, min_samples_leaf=10, criterion="friedman_mse", min_weight_fraction_leaf=0.008, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.3, max_depth=10, min_samples_split=6, min_samples_leaf=10, criterion="friedman_mse", min_weight_fraction_leaf=0.008, bootstrap=True, oob_score=True, random_state=rs)
                        
                        
                # #Toluene
                # elif var_of_int == 'Toluene (B)':
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.4, max_depth=9, min_samples_split=9, min_samples_leaf=16, criterion="absolute_error", min_weight_fraction_leaf=0.00, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #        PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.4, max_depth=8, min_samples_split=7, min_samples_leaf=9, criterion="absolute_error", min_weight_fraction_leaf=0.00, bootstrap=True, oob_score=True, random_state=rs) 
                        
                # # Isoprene
                # elif var_of_int == 'Isoprene (E_B)':
                #     if time == 'T0':
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=10, max_features=0.3, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                #     else:
                #         PVOCAL_EST = RandomForestRegressor(n_estimators=10, max_features=0.3, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="absolute_error", min_weight_fraction_leaf=0.01, bootstrap=True, oob_score=True, random_state=rs)
                
                # #Ethane
                elif var_of_int == 'Ethane (E)':
                    if time == 'T0':
                        PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.5, max_depth=9, min_samples_split=10, min_samples_leaf=10, criterion="friedman_mse", min_weight_fraction_leaf=0.013, bootstrap=True, oob_score=True, random_state=rs)
                    else:
                        PVOCAL_EST = RandomForestRegressor(n_estimators=200, max_features=0.5, max_depth=8, min_samples_split=9, min_samples_leaf=8, criterion="absolute_error", min_weight_fraction_leaf=0.014, bootstrap=True, oob_score=True, random_state=rs)
                # # PVOCAL_est = RandomForestRegressor(random_state=rs)
                # # elif var_of_int == 'Ethane (E)':

                # PVOCAL_EST = RandomForestRegressor(random_state=rs)
                
                # param_grid = {
                #             'n_estimators': [200], 
                #             'max_depth': [6,8,12],
                #             'max_features': [.3,.4],
                #             'min_samples_split': [2],
                #             'min_samples_leaf' : [1],
                #             'criterion' : ['absolute_error'],
                #             'min_weight_fraction_leaf' : [0.01],
                #             'bootstrap' : [True],
                #             'oob_score' : [True],
                #             'ccp_alpha' : [0.0]
                #             }
                
                
            elif model == 'GB':

                PVOCAL_EST = GradientBoostingRegressor(random_state=rs)
                
                param_grid = {
                            'loss': ['huber'],
                            # 'warm_start':[True],
                            # 'n_iter_no_change': [25],
                            # 'tol':[0.01],
                            'max_features': [.3,.5,.8],
                            'criterion' : ['friedman_mse'],
                            'min_weight_fraction_leaf' : [0.1],
                            # 'validation_fraction': [.1],
                            'subsample': [.3,.5,.8],
                            'n_estimators': [800],
                            'learning_rate': [0.01],
                            'max_depth': [3],
                            'alpha' : [0.9]
                            }
            else: 
                continue
                ## ADA Boost implementation

            # Create the GridSearchCV object
            # grid_search = GridSearchCV(estimator=PVOCAL_EST, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=12)
            
            # fit_params = {'sample_weight': samp_weights_train}
            
            # Fit the model to the training data
            # grid_search.fit(x_train, y_train.values.ravel())#,**fit_params)
            
            print(model)
            print(var_of_int)
            print(t)
            # bestParam = grid_search.best_params_
            # print(bestParam)
            
            # PVOCAL_EST = grid_search.best_estimator_
            
            # tempcsv = pd.DataFrame(grid_search.cv_results_)
            # tempcsv.to_csv("C:/Users/vwgei/Documents/PVOCAL/data/gridSearchCV_Results/GridSearchResults" + model + smog_label + var_of_int.split()[0] + t + ".csv", index=False)
            
            # print(PVOCAL_EST)
            dump(PVOCAL_EST, "C:/Users/vwgei/Documents/PVOCAL/bestmodels/" + model + smog_label + var_of_int.split()[0] + t + ".joblib")
            
            PVOCAL_EST.fit(x_train,y_train.values.ravel())
            
            # Make predictions on the training data
            y_pred_train = PVOCAL_EST.predict(x_train)
            # Make predictions on the testing data.
            y_pred_test = PVOCAL_EST.predict(x_test)


            # scores = cross_val_score(PVOCAL_EST, x_test, y_pred_test, cv=3)
            # print("Cross Validation Scores")
            # print(scores)
            
            if model == 'GB':
                # Compute the cumulative MSE on the test set at each stage
                test_errors = [mean_absolute_error(y_test, y_pred) for y_pred in PVOCAL_EST.staged_predict(x_test)]
                train_errors = [mean_absolute_error(y_train, y_pred) for y_pred in PVOCAL_EST.staged_predict(x_train)]
                
                # Plot the errors over boosting stages
                plt.plot(np.arange(1, len(test_errors) + 1), test_errors, label='Test Set MAE', color='orange')
                plt.plot(np.arange(1, len(train_errors) + 1), train_errors, label='Training Set MAE', color='blue')
                plt.xlabel('Boosting Iterations')
                plt.ylabel('Mean squared Error')
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
            # if model == 'RFR':
            #     custom_loss_train = custom_absolute_error(y_train, y_pred_train, sw_train)
            #     custom_loss_test = custom_absolute_error(y_test, y_pred_test, sw_test)
            
            

            # Calculate and organize metrics for training set
            train_metrics = calculate_and_organize_metrics(y_train, y_pred_train, REL_METHOD, REL_XTRM_TYPE, REL_COEF, smog)
            
            # Calculate and organize metrics for testing set
            test_metrics = calculate_and_organize_metrics(y_test, y_pred_test, REL_METHOD, REL_XTRM_TYPE, REL_COEF, smog)
            
            
            
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
            scatter_plot(y_train, y_pred_train, title, var_of_int, t, label, model, smog_label, "_Train", REL_METHOD, REL_XTRM_TYPE, REL_COEF, smog)

            # Visulize predictions on testing set
            title = 'Testing Results: ' + str.split(var_of_int)[0] + " " + time
            scatter_plot(y_test, y_pred_test, title, var_of_int, t, label, model, smog_label, "_Test", REL_METHOD, REL_XTRM_TYPE, REL_COEF, smog)
            
            
            # Generate permutation importances using function
            perm_importances, perm_importances_index = plot_permutation_importance_scores(PVOCAL_EST, x_train, y_train, model, smog_label, title="Permutation Importances using Train: " + var_of_int.split()[0] + " " + time)
            
            
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
                        
            # # One-Way Partial Dependence
            # # Display the partial dependence plot
            # display = PartialDependenceDisplay.from_estimator(PVOCAL_EST,
            #                                                     X=x_train,
            #                                                     features=top_two_indices),
            #                                                     # kind="individual")
            
            # if time == 'T0':
            #     latitude = "Latitude_0"
            #     longitude = "Longitude_0"
            #     pt = "Potential_Temperature_0"
            #     sr = "Solar_Radiation_0"
            # else:
            #     latitude = "Latitude_-24"
            #     longitude = "Longitude_-24"
            #     pt = "Potential_Temperature_-24"
            #     sr = "Solar_Radiation_-24"
            
            # #From SKlearn PDP tutorial!   
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
            plt.clf()



