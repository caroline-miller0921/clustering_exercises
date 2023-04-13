import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
import wrangle

import warnings
warnings.filterwarnings('ignore')

def plot_model_preds(X_train, feature, y, yhat, target):

    plt.scatter(X_train[[f'{feature}']], y, color='magenta')

    plt.plot(X_train[[f'{feature}']], yhat, color='black')
    plt.title(f'Linear Regression Model (feature = {feature})')
    plt.xlabel(f'{feature}')
    plt.ylabel(f'Target variable: {target}')
    plt.show()


def plot_residuals(y, yhat):
    '''
    input the array or series that contains the actual y values as y
    input the array or series that contains the y predictions of the model as yhat
    '''
    plt.scatter(yhat, (y - yhat), color='purple')

    plt.title('Linear Regression Model Residual')
    plt.xlabel('Predictions')
    plt.ylabel('Residual')
    plt.show()

def regression_errors(y, yhat):

    SSE = mean_squared_error(y, yhat)*len(y)
    ESS = sum((yhat - (y.mean())) ** 2)
    TSS = SSE + ESS
    MSE = mean_squared_error(y, yhat)
    RMSE = sqrt(MSE)

    regression_errors_df = pd.DataFrame(
    {
        'model_metrics': [SSE, ESS, TSS, MSE, RMSE]
        }, index = ['SSE', 'ESS', 'TSS', 'MSE', 'RMSE']
    )
    
    return regression_errors_df

def baseline_mean_errors(y):
    baseline = [[y.mean()]] * len(y)
    
    SSE_baseline = mean_squared_error(y, baseline)*len(y)
    MSE_baseline = mean_squared_error(y, baseline)
    RMSE_baseline = sqrt(MSE_baseline)
    
    baseline_errors_df = pd.DataFrame(
    {
        'baseline_metrics': [SSE_baseline, MSE_baseline, RMSE_baseline]
        }, index = ['SSE', 'MSE', 'RMSE']
    )
    
    return baseline_errors_df

def better_than_baseline(y, yhat):  
    
    SSE = mean_squared_error(y, yhat)*len(y)
    MSE = mean_squared_error(y, yhat)
    RMSE = sqrt(MSE)
    
    baseline = [[y.mean()]] * len(y)
    
    SSE_baseline = mean_squared_error(y, baseline)*len(y)
    MSE_baseline = mean_squared_error(y, baseline)
    RMSE_baseline = sqrt(MSE_baseline)
    
    model_comparison_df = pd.DataFrame(
        {
        'baseline_metrics': [SSE_baseline, MSE_baseline, RMSE_baseline],
        'model_metrics': [SSE, MSE, RMSE]
        }, index = ['SSE', 'MSE', 'RMSE']
    )
    
    if SSE < SSE_baseline:
        print(f'The model SSE is less than the baseline SSE. This indicates that the model performs better than the baseline.\n\n')
    else:
        print(f'The baseline SSE is less than or equal to the model SSE. This indicates that the baseline performs better than the model.\n\n')
        
    if MSE < MSE_baseline:
        print(f'The model MSE is less than the baseline MSE. This indicates that the model performs better than the baseline.\n\n')
    else:
        print(f'The baseline MSE is less than or equal to the model MSE. This indicates that the baseline performs better than the model.\n\n')
        
    if RMSE < RMSE_baseline:
        print(f'The model RMSE is less than the baseline RMSE. This indicates that the model performs better than the baseline.\n\n')
    else:
        print(f'The baseline RMSE is less than or equal to the model RMSE. This indicates that the baseline performs better than the model.\n\n')
        
    print(f'\nIs the model better than the baseline?\n')
    if SSE < SSE_baseline and MSE < MSE_baseline and RMSE < RMSE_baseline:
        print('True')
    else:
        print('False')
    
    return model_comparison_df