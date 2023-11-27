# imports: 
import numpy as np
import pandas as pd
import os 

# spliting data
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import MinMaxScaler
    


#--------------- Data Acqusition -------------------#
def wine_df ():
    '''
    This function will:
    - read in two csv files: winequality-red.csv and winequlaity-white.csv
    - it will then create a new column wine_type
    - it will combined the two data frames into a new wine_df
    - it will read the dataframe into a new csv: wine.csv
    '''
    
    # create the csv: 
    if os.path.isfile('wine.csv'):
        
        #if the csv file exists it will read the file
        wine_df = pd.read_csv('wine.csv', index_col = 0)
    
    else: 
        # create a table for red wine
        red = pd.read_csv('winequality-red.csv')
    
        # create a table for white wine
        white = pd.read_csv('winequality-white.csv')
    
        # create columns in both tables that describe what the wine type is 
        red['wine_color'] = 'red'
        white['wine_color'] = 'white'
    
        # combined the two data frames
        wine_df = pd.concat([red, white], ignore_index = True)
    
        wine_df.to_csv('wine.csv')
        
    return wine_df

#----------------Data Preperation------------------#
def identify_columns(df):
    cat_cols, num_cols = [], []

    for col in df.columns:
        if df[col].dtype == 'O':
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else:
                num_cols.append(col)

    return cat_cols, num_cols



#remove outliers function
def remove_outliers(df, k):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    cat_cols, num_cols = identify_columns(df)
    
    for col in num_cols:
        
        # For each column, it calculates the first quartile (q1) and 
        #third quartile (q3) using the .quantile() method, where q1 
        #corresponds to the 25th percentile and q3 corresponds to the 75th percentile.
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def identify_columns(df):
    cat_cols, num_cols = [], []

    for col in df.columns:
        if df[col].dtype == 'O':
            cat_cols.append(col)
        else:
            if df[col].nunique() < 10:
                cat_cols.append(col)
            else:
                num_cols.append(col)

    return cat_cols, num_cols

def clean_wine(df):
    '''
    This function will: 
    - remove outliers:
    - change columns names to have underscores
    - make the data types the same
    - encode 1 for red and 0 for white
    '''
    
    
    # change the names
    df.columns = df.columns.str.replace(' ','_')
    
    
    #encoding 
    df['is_white'] = df['wine_color'].map({'red': 1, 'white': 0})
    
    # change data types:
    df['quality'] = df['quality'].astype(float)
    df['is_white'] = df['is_white'].astype(float)
    
    return df





#----------Split Funciton--------#
def split_data(df, target, seed=123):
    '''
    This function takes in a dataframe and splits the data into train, validate and test. 
    - The test data set is 20
    - Then the data frame is split 70% train, 30% validate
    '''
    train_validate, test = train_test_split(df, test_size=0.2, random_state=seed)
    
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=seed)
    return train, validate, test




#----------Min Max Scaler--------#
def scale_data(train, 
               validate, 
               test, 
               to_scale):
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled