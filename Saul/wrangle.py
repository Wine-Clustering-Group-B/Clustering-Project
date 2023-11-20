import numpy as np
import pandas as pd
import os
import env
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

directory = os.getcwd()
# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import for scaling data
from sklearn.preprocessing import MinMaxScaler

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")


# imports for modeling:


#------------------------------------------------------------------------------
def get_connection_url(db, user=env.user, host=env.host, password=env.password):
    """
    This function will:
    - take username, pswd, host credentials from imported env module
    - output a formatted connection_url to access mySQL db
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
#------------------------------------------------------------------------------
SQL_query = '''
SELECT prop.*, 
       pred.logerror, 
       pred.transactiondate, 
       air.airconditioningdesc, 
       arch.architecturalstyledesc, 
       build.buildingclassdesc, 
       heat.heatingorsystemdesc, 
       landuse.propertylandusedesc, 
       story.storydesc, 
       construct.typeconstructiondesc 

FROM   properties_2017 prop  
       INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid, logerror) pred
               USING (parcelid) 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
WHERE  prop.latitude IS NOT NULL 
       AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31'
'''
#------------------------------------------------------------------------------
def new_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the zillow db
    """
    url = get_connection_url('zillow')
    
    return pd.read_sql(SQL_query, url)
#------------------------------------------------------------------------------

def get_data(SQL_query, directory, filename = 'zillow.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv
    - outputs zillow df
    """
    if os.path.exists('./zillow.csv'): 
        return pd.read_csv('zillow.csv')
        return df
    else:
        df = new_data(SQL_query)

        df.to_csv(filename)
        return df
#------------------------------------------------------------------------------
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows': num_missing, 'percent_rows_missing': pct_missing})
    return cols_missing


#------------------------------------------------------------------------------        
def summary_of_data(df):
    print('--------------------------------')
    print('--------------------------------')
    print('Information on DataFrame: ')
    print(f'Shape of Dataframe: {df.shape}')
    print('--------------------------------')
    print(f'Basic DataFrame info:')
    print(df.info())
    print('--------------------------------')
    print('Missing Values by Column: ')
    print(df.isna().sum())
    print('Missing Values by Row: ')
    print(missing_in_row(df).to_markdown())
    print('--------------------------------')
    print('--------------------------------')
#------------------------------------------------------------------------------
def prepare_data(df):
    '''
    prepare will take in zillow data, remove any whitespace values
    drop out null values
    and return the entire dataframe.
    '''
    #drop null values
    df = df.dropna()
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df

#------------------------------------------------------------------------------
def split_data(df):
    '''
    This function performs split on df data.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
                                        
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test
#------------------------------------------------------------------------------
def missing_in_row(df):
    return pd.concat(
        [
            df.isna().sum(axis=1),
            (df.isna().sum(axis=1) / df.shape[1])
        ], axis=1).rename(
        columns={0:'missing_cells', 1:'percent_missing'}
    ).groupby(
        ['missing_cells',
         'percent_missing']
    ).count().reset_index().rename(columns = {'index': 'num_missing'})
#------------------------------------------------------------------------------
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df
#------------------------------------------------------------------------------
def min_max_scaler(train, validate, test):
    """
    This function will:
    - take in train, validate, and test data
    - scale columns
    - return a df of scaled data
    """
    import sklearn
    from sklearn.preprocessing import MinMaxScaler
    
    #Create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    validate[num_vars] = scaler.transform(validate[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, validate, test
#------------------------------------------------------------------------------    
cols_to_remove = ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc', 'unitcnt', 'Unnamed: 0']
    
def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df


#------------------------------------------------------------------------------
# Function to read and wrangle data:

def wrangle_zillow(df):
    
    # Restrict df to only properties that meet single unit use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc','heatingorsystemdesc','unitcnt'
                            ,'buildingqualitytypeid'])


    # replace nulls in unitcnt with 1
#     df.unitcnt.fillna(1, inplace = True)
    
    # assume that since this is Southern CA, null means 'None' for heating system
#     df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
#     df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    
    return df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
