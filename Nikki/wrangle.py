# imports: 
import pandas as pd
import numpy as np
import os


#---------------------------Acquire Functions--------------------#
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
        red['wine_type'] = 'red'
        white['wine_type'] = 'white'
    
        # combined the two data frames
        wine_df = pd.concat([red, white], ignore_index = True)
    
        wine_df.to_csv('wine.csv')
        
    return wine_df



#----------------Outliers--------------#
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:
        
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



#----------------Clean-----------------#
def clean_wine(df):
    '''
    This will rename the columns in the wine data base
    '''
    df.columns = df.columns.str.replace(' ','_')
    
    return df

    
    
#-----------------Identify columns----------------#
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