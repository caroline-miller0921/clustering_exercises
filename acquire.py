# inside acquire.py script:
from env import username as u, password as p, host as h, url
import pandas as pd
import os
db = ''

def get_sql_url(schema, db, u=u, p=p, h=h):
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''
    url = f'mysql+pymysql://{u}:{p}@{h}/{schema}'
    return pd.read_sql(f'select * from {db};', url)

def get_connection(schema, u=u, p=p, h=h):
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''
    return f'mysql+pymysql://{u}:{p}@{h}/{schema}'

def get_zillow_2017():
    '''
    get_sql_url will pull the credentials present from any current env
    file in the same directory as this acquire script
    and will return a connection based on what schema and databases 
    (db) are handed to the function call
    '''

    query ='''
    select * from 
        (
            select * from 
            properties_2017 
            left join propertylandusetype 
            using (propertylandusetypeid)
            left join airconditioningtype
            using (airconditioningtypeid) 
            left join architecturalstyletype
            using (architecturalstyletypeid)
            left join buildingclasstype
            using (buildingclasstypeid)
            left join heatingorsystemtype
            using(heatingorsystemtypeid)
            left join storytype
            using (storytypeid)
            left join typeconstructiontype
            using (typeconstructiontypeid)
        ) as z
        join predictions_2017 as y using (parcelid);
    '''
    
    if os.path.isfile('zillow_2017.csv'):

        df = pd.read_csv('zillow_2017.csv')
    
    else:

        url = f'mysql+pymysql://{u}:{p}@{h}/zillow'

        df = pd.read_sql(f'{query}', url)

        df.to_csv('zillow_2017.csv')
        
    query2 ='''
    select * from unique_properties;
    '''
    
    if os.path.isfile('unique_properties.csv'):

        unique_df = pd.read_csv('unique_properties.csv')
    
    else:

        url = f'mysql+pymysql://{u}:{p}@{h}/zillow'

        unique_df = pd.read_sql(f'{query2}', url)

        unique_df.to_csv('unique_properties.csv')
        
    unique_df.drop(columns='Unnamed: 0', inplace=True)
    unique_df['unique'] = 'unique'
    
    zillow = df.merge(unique_df, how='left', on='parcelid')
    
    zillow.drop(columns='Unnamed: 0', inplace=True)
    
    zillow = zillow.sort_values(by='transactiondate', ascending=False)
    
    zillow = zillow.drop_duplicates(subset=['parcelid'], keep='first')
    
    zillow = zillow[(zillow['latitude'].isna() == False) | (zillow['longitude'].isna() == False)]
    
    return zillow
