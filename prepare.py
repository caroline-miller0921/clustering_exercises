import env
import acquire
import pandas as pd
from sklearn.model_selection import train_test_split

def prep_titanic():
    '''
    clean titanic will take in a single pandas dataframe
    and will proceed to drop redundant columns
    and nonuseful information
    in addition to addressing null values
    and encoding categorical variables
    '''
    df = acquire.get_titanic_data()
    df = df.drop(columns=['passenger_id', 'embarked', 'class', 'deck'])
    df['age'] = df['age'].fillna(df.age.mean())
    # encode categorical values
    df = pd.concat(
        [df, pd.get_dummies(df[['sex', 'embark_town']], 
                            drop_first=True)], axis=1)
    df = df.drop(columns=['sex', 'embark_town'])
    return df

def prep_iris():
    '''
    clean iris will take in a single pandas dataframe
    and will proceed to drop redundant columns
    and nonuseful information
    in addition to addressing null values
    and encoding categorical variables
    '''
    df = acquire.get_iris_data()
    # take out redundent comlumns
    df = df.drop(columns=['species_id', 'measurement_id'])
    # rename species_name to just species
    df = df.rename(columns={'species_name': 'species'})
    # encode categorical values
    df = pd.concat(
        [df, pd.get_dummies(df[['species']], 
                            drop_first=True)], axis=1)
    return df

def prep_telco():
    '''
    prep telco will take in a single pandas dataframe
    and will proceed to drop redundant columns
    and nonuseful information
    in addition to addressing null values
    and encoding categorical variables
    '''
    df = acquire.get_telco_data()
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    df = df.replace('Yes', 1).replace('No', 0)
    df = df.replace('No phone service', 0)
    df = df.replace('No internet service', 0)
    df['total_charges'] = df.total_charges.replace(' ', 0.0)
    df['total_charges'] = df.total_charges.astype(float)
    df = pd.concat(
        [df, pd.get_dummies(df[['gender', 'contract_type', 'internet_service_type', 'payment_type']], 
                            drop_first=True)], axis=1)
    return df

def split_data(df, target):
    '''
    split data takes in a dataframe or function which returns a dataframe
    and will split data based on the values present in a cleaned 
    version of the dataframe. Also you must provide the target
    at which you'd like the stratify (a feature in the DF)
    '''
    train_val, test = train_test_split(df, 
                                       train_size=.8,
                                       random_state=1349, 
                                       stratify=df[target])
    train, validate = train_test_split(train_val, 
                                       train_size=0.7,
                                       random_state=1349,
                                       stratify=train_val[target])
    return train, test, validate

