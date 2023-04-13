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

def single_family(df):

    df = df[(df['propertylandusedesc'] != 'Duplex (2 Units, Any Combination)') & (df['propertylandusedesc'] != 'Planned Unit Development') & (df['propertylandusedesc'] != 'Quadruplex (4 Units, Any Combination)') & (df['propertylandusedesc'] != 'Triplex (3 Units, Any Combination)') & (df['propertylandusedesc'] != 'Cluster Home') & (df['propertylandusedesc'] != 'Commercial/Office/Residential Mixed Used')]
    return df

def remove_columns(df, cols_to_remove):
    '''
    This function takes in a dataframe 
    and the columns that need to be dropped
    then returns the desired dataframe.
    '''
    df = df.drop(columns=cols_to_remove)
    print(f'Columns dropped: {cols_to_remove}')
    return df

def remove_ids(df):
    cols = df.columns.to_list()
    id_cols = []
    for col in cols:
        r = 'id'
        if r in col:
            id_cols.append(col)
            df = df.drop(columns=col)
    print(f'Columns dropped: {id_cols}')
    return df

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''
    This function takes in a dataframe, the percent of columns and rows
    that need to have values/non-nulls
    and returns the dataframe with the desired amount of nulls left.

    prop_required_columns:  
    A number between 0 and 1 that represents the proportion, for each column, of rows with 
    non-missing values required to keep the column. i.e. if prop_required_column = .6, then 
    you are requiring a column to have at least 60% of values not-NA (no more than 40% missing).

    prop_required_rows:
    A number between 0 and 1 that represents the proportion, for each row, of columns/variables 
    with non-missing values required to keep the row. For example, if prop_required_row = .75, 
    then you are requiring a row to have at least 75% of variables with a non-missing value 
    (no more that 25% missing).
    '''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df

def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    '''
    This function uses two other functions to remove columns 
    and desired number of nulls values
    then returns the cleaned dataframe with acceptable number of nulls.
    '''
    df = single_family(df)
    df = remove_ids(df)
    df = remove_columns(df, col_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df

# def get_upper_outliers(s, k=1.5):
#     '''
#     Given a series and a cutoff value, k, returns the upper outliers for the
#     series.

#     The values returned will be either 0 (if the point is not an outlier), or a
#     number that indicates how far away from the upper bound the observation is.
#     '''
#     q1, q3 = s.quantile([.25, 0.75])
#     iqr = q3 - q1
#     upper_bound = q3 + k * iqr
#     return s.apply(lambda x: max([x - upper_bound, 0]))

# def add_upper_outlier_columns(df, k=1.5):
#     '''
#     Add a column with the suffix _outliers for all the numeric columns
#     in the given dataframe.
#     '''
#     for col in df.select_dtypes('number'):
#         df[col + '_outliers_upper'] = get_upper_outliers(df[col], k)
#     return df

# def identfy_upper_outliers(df):
#     outlier_cols = [col for col in df.columns if col.endswith('_outliers_upper')]
#     for col in outlier_cols:
#         print(col, ': ')
#         subset = df[col][df[col] > 0]
#         print(f'Number of Observations Above Upper Bound: {subset.count()}', '\n')
#         print(subset.describe())
#         print('------', '\n')
#     for col in outlier_cols:
#         sns.histplot(data=df, x=col, palette='crest')
#         plt.title(f'Distribution of {col}')
#         plt.ylabel('Count of Properties')
#         plt.xlabel(f'{col}')
#     for col in outlier_cols:
        
# def remove_outliers(df, df_cols, k=1.5):
#     num_cols = {}
    
#     for col in df.select_dtypes('number'):
#         num_cols[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    
#     for col in df_cols:
#         col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
#         # print(col_qs)
    
#     for col in df_cols:    
#         iqr = col_qs[col][0.75] - col_qs[col][0.25]
#         lower_fence = col_qs[col][0.25] - (iqr*k)
#         upper_fence = col_qs[col][0.75] + (iqr*k)
#         #print(f'Lower fence of {col}: {lower_fence}')
#         #print(f'Upper fence of {col}: {upper_fence}')
#         df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
#     return df

def remove_outliers(df, k=1.5):
    
    col_qs = {}
    num_cols = []
    for col in df.select_dtypes('number'):
        num_cols.append(col)
    # print(num_cols)
    for col in num_cols:
        # num_cols = np.where(df[col].value_counts()) > 10)
        if df[col].nunique() < 10:
            num_cols.remove(col) 
    
    # print(num_cols)

    for col in num_cols:
        try:
            col_qs[col] = df[col].quantile([0.25, 0.75])
        except Exception as e:
            print(e)
            print(f'Problem column: {col}')
            num_cols.remove(col)
    num_cols.remove('roomcnt')
    num_cols.remove('longitude')
    num_cols.remove('latitude')
    num_cols.remove('censustractandblock')
    num_cols.remove('logerror')
    
    for col in num_cols:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (iqr*k)
        upper_fence = col_qs[col][0.75] + (iqr*k)
        print(f'Lower fence of {col}: {lower_fence}')
        print(f'Upper fence of {col}: {upper_fence}')
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

# Function to remove or impute Null values

def impute_nulls(df):
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df[['unitcnt']])
    df[['unitcnt']] = imputer.transform(df[['unitcnt']])
    df['unitcnt'] = df['unitcnt'].astype(float)

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df[['heatingorsystemdesc']])
    df['heatingorsystemdesc'] = imputer.transform(df[['heatingorsystemdesc']])
    df['heatingorsystemdesc'] = df['heatingorsystemdesc'].astype(str)

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df[['censustractandblock']])
    df[['censustractandblock']] = imputer.transform(df[['censustractandblock']])
    df['censustractandblock'] = df['censustractandblock'].astype(float)

    return df

def bivariate_visulization(df, target):
    
    cat_cols, num_cols = [], []
    
    for col in df.columns:
        if df[col].dtype == "o":
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else: 
                num_cols.append(col)

    num_cols.remove('transactiondate')
    num_cols.remove('censustractandblock')

    cat_cols.remove('assessmentyear')

    print(f'Numeric Columns: {num_cols}')
    print(f'Categorical Columns: {cat_cols}')
    explore_cols = cat_cols + num_cols

    for col in explore_cols:
        if col in cat_cols:
            if col != target:
                print(f'Bivariate assessment of feature {col}:')
                sns.barplot(data = df, x = df[col], y = df[target], palette='crest')
                plt.show()

        if col in num_cols:
            if col != target:
                print(f'Bivariate feature analysis of feature {col}: ')
                plt.scatter(x = df[col], y = df[target], color='turquoise')
                plt.axhline(df[target].mean(), ls=':', color='red')
                plt.axvline(df[col].mean(), ls=':', color='red')
                plt.show()

    print('_____________________________________________________')
    print('_____________________________________________________')
    print()

def univariate_visulization(df):
    
    cat_cols, num_cols = [], []
    for col in df.columns:
        if df[col].dtype == "o":
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else: 
                num_cols.append(col)
    num_cols.remove('transactiondate')
    num_cols.remove('censustractandblock')
    cat_cols.remove('assessmentyear')
                
    explore_cols = cat_cols + num_cols
    print(f'cat_cols: {cat_cols}')
    print(f'num_cols: {num_cols}')
    for col in explore_cols:
        
        if col in cat_cols:
            print(f'Univariate assessment of feature {col}:')
            sns.countplot(data=df, x=col, color='turquoise', edgecolor='black')
            plt.show()

        if col in num_cols:
            print(f'Univariate feature analysis of feature {col}: ')
            plt.hist(df[col], color='turquoise', edgecolor='black')
            plt.show()
            df[col].describe()
    print('_____________________________________________________')
    print('_____________________________________________________')
    print()

def viz_explore(df, target):

    univariate_visulization(df)

    bivariate_visulization(df, target)

    num_cols = []
    for col in df.select_dtypes('number'):
        num_cols.append(col)
    
    print(num_cols)

    num_cols.remove('assessmentyear')
    num_cols.remove('censustractandblock')  
    num_cols.remove('longitude')
    num_cols.remove('latitude')

    print(num_cols)

    zillow_corr = df[num_cols].corr(method='spearman')
    plt.figure(figsize=(30,25))
    sns.set(font_scale=2.2) 
    sns.heatmap(zillow_corr, cmap='crest', annot=True, mask=np.triu(zillow_corr))
    plt.show()

# ------------------ Splitting Data -------------------

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

# ------------------ Scaling --------------------------

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
    
def rfe(X, y, n):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n)
    rfe.fit(X, y)
    
    rfe_df = pd.DataFrame(
    {
        'feature_ranking': [*rfe.ranking_],
        'selected': [*rfe.get_support()]
    }, index = X.columns
    )
    
    cols = []
    
    cols = [*X.columns[rfe.get_support()]]
    
    print(f'The {n} features selected are as follows:\n {cols}')
    
    return rfe_df

def select_kbest(X, y, k):
    kbest =  SelectKBest(f_regression, k=k)
    
    _ = kbest.fit(X, y)
    
    kbest_df = pd.DataFrame(
    {
        'statistical_f_values': [*kbest.scores_],
        'p_values': [*kbest.pvalues_],
        'selected': [*kbest.get_support()]
    }, index = X.columns
    )
    
    cols = []
    
    cols = [*X.columns[kbest.get_support()]]
    
    print(f'The features selected with the k value set to {k} are as follows:\n {cols}')
    
    return kbest_df