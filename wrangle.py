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

def get_zillow_2017():
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''

    if os.path.isfile('zillow_2017.csv'):

        return pd.read_csv('zillow_2017.csv')
    
    else:

        url = f'mysql+pymysql://{u}:{p}@{h}/zillow'

        df = pd.read_sql(f'select parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid, transactiondate from (select * from properties_2017 left join propertylandusetype using (propertylandusetypeid) where propertylandusetypeid <=> 261 or propertylandusetypeid <=> 279) as p join predictions_2017 as y using (parcelid);', url)

        df.to_csv('zillow_2017.csv')

        return df

def remove_outliers(df, k=1.5):
    col_qs = {}
    
    for col in df.select_dtypes('number'):
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
        # print(col_qs)
    
    for col in df.select_dtypes('number'):    
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (iqr*k)
        upper_fence = col_qs[col][0.75] + (iqr*k)
        #print(f'Lower fence of {col}: {lower_fence}')
        #print(f'Upper fence of {col}: {upper_fence}')
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df


def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'fips_location', 'propertytypeid', 'transactiondate']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        colors = ['turquoise']

        # Display histogram for column.
        plt.hist(x=df[col], bins=5 , color = colors, edgecolor = 'black')

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()


def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'squarefeet', 'tax_value', 'taxamount', 'yearbuilt', 'property_age']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]], palette='crest')

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()


def get_count(df):

    plt.figure(figsize=(16, 10))

    sns.countplot(data=df, x='fips_location', palette='crest')
    plt.xlabel('County Name')
    plt.ylabel('Count (mil)')
    plt.title(f'Propery Count by FIPS Codes')
    plt.show()


def prepare_zillow(df):

    df = pd.read_csv('zillow_2017.csv')

    df = df.drop(columns='parcelid')

    df['transactiondate'] = pd.to_datetime(df['transactiondate'])

    # dropping all rows that have 0 bedrooms or 0 bathrooms
    df = df[(df.bedroomcnt != 0) | (df.bathroomcnt != 0)]

    df.rename(columns={'bedroomcnt': 'bedrooms',
                        'bathroomcnt': 'bathrooms', 
                        'calculatedfinishedsquarefeet': 'squarefeet', 
                        'taxvaluedollarcnt': 'tax_value', 
                        'propertylandusetypeid': 'propertytypeid'}, 
                        inplace=True)
    
    df = df.drop(columns=('Unnamed: 0'))

    df = remove_outliers(df, ['bedrooms', 'bathrooms', 'squarefeet', 'tax_value', 'taxamount'])

    df['fips_location'] = df.fips
    df.fips_location = df.fips_location.replace(6037.0, 'Los Angeles County, CA')
    df.fips_location = df.fips_location.replace(6059.0, 'Orange County, CA')
    df.fips_location = df.fips_location.replace(6111.0, 'Colusa County, CA')

    df.fips = df.fips.astype(int)

    df.propertytypeid = df.propertytypeid.astype(int)

    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['yearbuilt']])

    train[['yearbuilt']] = imputer.transform(train[['yearbuilt']])
    validate[['yearbuilt']] = imputer.transform(validate[['yearbuilt']])
    test[['yearbuilt']] = imputer.transform(test[['yearbuilt']])  
    
    train['yearbuilt'] = train['yearbuilt'].astype(int)
    validate['yearbuilt'] = validate['yearbuilt'].astype(int)
    test['yearbuilt'] = test['yearbuilt'].astype(int)

    train['property_age'] = 2017 - train['yearbuilt']
    validate['property_age'] = 2017 - validate['yearbuilt']
    test['property_age'] = 2017 - test['yearbuilt']

    get_hist(train)
    get_box(train)
    get_count(train)

    return train, validate, test

def wrangle_zillow():

    train, validate, test = prepare_zillow(get_zillow_2017())

    return train, validate, test

def scale_and_vis(scaler, X, columns):
    X_scaled = X.copy()
    X_scaled[columns] = scaler.fit_transform(X_scaled[columns])

    fig, axs = plt.subplots(len(columns), 2, figsize=(16, 11))
    for (ax1, ax2), col in zip(axs, columns):

        ax1.hist(X[col], color='turquoise')
        ax1.set(title=f'Distribution of Unscaled {col}', xlabel=f'Value of {col}', ylabel=f'Count of {col}')

        ax2.hist(X_scaled[col], color='turquoise')
        ax2.set(title=f'Distribution of Scaled {col}', xlabel=f'Value of {col}', ylabel=f'Count of {col}')
    
    fig.suptitle(f'Scaling Visualization for {scaler}', fontsize=16)
    
    plt.tight_layout()
    plt.show()

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
            if df[col].nunique() < 5:
                cat_cols.append(col)
            else: 
                num_cols.append(col)
                
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

def viz_explore(train, target):

    univariate_visulization(train)

    bivariate_visulization(train, target)

    zillow_corr = train.drop(columns={'transactiondate', 'propertytypeid'}).corr(method='spearman')
    sns.heatmap(zillow_corr, cmap='crest', annot=True, mask=np.triu(zillow_corr))
    plt.show()

