import pandas as pd
from IPython.display import display

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