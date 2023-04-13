from env import username as u, password as p, host as h, url
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import numpy as np
from IPython.display import display

def get_mall_customers():
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''

    query ='''
    select * from customers;
    '''
    
    if os.path.isfile('mall_customers.csv'):

        df = pd.read_csv('mall_customers.csv')
    
    else:

        url = f'mysql+pymysql://{u}:{p}@{h}/mall_customers'

        df = pd.read_sql(f'{query}', url)

        df.to_csv('mall_customers.csv')
    
    df = df.drop(columns='Unnamed: 0')
    
    return df

def nulls_by_row(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values in a row
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def nulls_by_col(df):
    '''
    This function takes in a dataframe 
    and finds the number of missing values
    it returns a new dataframe with quantity and percent of missing values
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('                    SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    display(pd.DataFrame(df.head(3)))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    display(pd.DataFrame(df.info()))
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    display(pd.DataFrame(df.describe().T))
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            display(pd.DataFrame(df[col].value_counts()))
        else:
            display(pd.DataFrame(df[col].value_counts(bins=10, sort=False)))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    display(pd.DataFrame(nulls_by_col(df)))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    display(pd.DataFrame(nulls_by_row(df)))
    print('=====================================================')

def remove_outliers(df, k=1.5):
    col_qs = {}
    
    df_cols = df.columns
    df_cols = df_cols.to_list()
    df_cols.remove('customer_id')
    df_cols.remove('gender')

    for col in df_cols:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
        # print(col_qs)
    
    for col in df_cols:    
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (iqr*k)
        upper_fence = col_qs[col][0.75] + (iqr*k)
        #print(f'Lower fence of {col}: {lower_fence}')
        #print(f'Upper fence of {col}: {upper_fence}')
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

def split_data(df):
    '''
    split data takes in a dataframe or function which returns a dataframe
    and will split data based on the values present in a cleaned 
    version of the dataframe. Also you must provide the target
    at which you'd like the stratify (a feature in the DF)
    '''
    train_val, test = train_test_split(df, 
                                       train_size=.8,
                                       random_state=1349)
    train, validate = train_test_split(train_val, 
                                       train_size=0.7,
                                       random_state=1349)
    return train, test, validate

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values, 
                                                  index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

def get_dummies(df):
    df = pd.concat(
        [df, pd.get_dummies(df[['gender']], 
                            drop_first=True)], axis=1)
    df = df.drop(columns=['gender', 'customer_id'])
    return df

def wrangle_mall():
    
    '''
    df_cols are the columns which are fed into the remove_outliers function.
    These columns should NOT contain categorical columns to include discrete numeric columns.
    '''

    # acquire the DF from MySQL
    df = get_mall_customers()

    # summarize the DF
    summarize(df)

    # Detect and drop outiers
    df = remove_outliers(df, k=1.5)

    # Encode categorical columns using a one hot encoder
    df = get_dummies(df)

    # Split the data
    train, validate, test = split_data(df)

    return train, validate, test


