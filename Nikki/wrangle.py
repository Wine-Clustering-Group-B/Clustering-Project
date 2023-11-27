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