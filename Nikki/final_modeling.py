# imports: 
 
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# clustering
from sklearn.cluster import KMeans

#regresion: 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector

#statistical metrics:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#-------------------Clustering Functions--------------#

def cluster_1(train, validate, test, features=['alcohol', 'sulphates'], n_clusters=4):
    """
    Cluster the data using KMeans on the specified features.

    Parameters:
    - train: Training data (DataFrame)
    - validate: Validation data (DataFrame)
    - test: Test data (DataFrame)
    - features: Features to use for clustering
    - n_clusters: Number of clusters for KMeans (default is 4)

    Returns:
    - train: Training data with added 'cluster_1' column
    - validate: Validation data with added 'cluster_1' column
    - test: Test data with added 'cluster_1' column
    """
    # Train KMeans on training data
    X1 = train[features]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X1)

    # Add 'cluster_1' column to training data
    train['cluster_1'] = kmeans.predict(X1)

    # Add 'cluster_1' column to validation data
    V1 = validate[features]
    validate['cluster_1'] = kmeans.predict(V1)

    # Add 'cluster_1' column to test data
    T1 = test[features]
    test['cluster_1'] = kmeans.predict(T1)
    
    # visualize the clusters: 
    sns.scatterplot(x = 'alcohol', y = 'sulphates', data = train, hue = 'cluster_1')
    plt.show()
    
    return train, validate, test




def cluster_2(train, validate, test, features=['free_sulfur_dioxide', 'total_sulfur_dioxide'], n_clusters=4):
    """
    Cluster the data using KMeans on the specified features.

    Parameters:
    - train: Training data (DataFrame)
    - validate: Validation data (DataFrame)
    - test: Test data (DataFrame)
    - features: Features to use for clustering 
    - n_clusters: Number of clusters for KMeans (default is 4)

    Returns:
    - train: Training data with added 'cluster_1' column
    - validate: Validation data with added 'cluster_1' column
    - test: Test data with added 'cluster_1' column
    """
    # Train KMeans on training data
    X2 = train[features]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X2)

    # Add 'cluster_1' column to training data
    train['cluster_2'] = kmeans.predict(X2)

    # Add 'cluster_1' column to validation data
    V2 = validate[features]
    validate['cluster_2'] = kmeans.predict(V2)

    # Add 'cluster_1' column to test data
    T2 = test[features]
    test['cluster_2'] = kmeans.predict(T2)
    

    # visualize the clusters: 
    sns.scatterplot(x = 'free_sulfur_dioxide', y = 'total_sulfur_dioxide', data = train, hue = 'cluster_2')
    plt.show()
    
    return train, validate, test


def cluster_3(train, validate, test, features=['volatile_acidity', 'density'], n_clusters=4):
    """
    Cluster the data using KMeans on the specified features.

    Parameters:
    - train: Training data (DataFrame)
    - validate: Validation data (DataFrame)
    - test: Test data (DataFrame)
    - features: Features to use for clustering 
    - n_clusters: Number of clusters for KMeans (default is 4)

    Returns:
    - train: Training data with added 'cluster_1' column
    - validate: Validation data with added 'cluster_1' column
    - test: Test data with added 'cluster_1' column
    """
    # Train KMeans on training data
    X3 = train[features]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X3)

    # Add 'cluster_1' column to training data
    train['cluster_3'] = kmeans.predict(X3)

    # Add 'cluster_1' column to validation data
    V3 = validate[features]
    validate['cluster_3'] = kmeans.predict(V3)

    # Add 'cluster_1' column to test data
    T3 = test[features]
    test['cluster_3'] = kmeans.predict(T3)
    
    # visualize the clusters: 
    sns.scatterplot(x = 'volatile_acidity', y = 'density', data = train, hue = 'cluster_3')
    plt.show()
    
    return train, validate, test



def cluster_4(train, validate, test, features=['citric_acid', 'pH'], n_clusters=4):
    """
    Cluster the data using KMeans on the specified features.

    Parameters:
    - train: Training data (DataFrame)
    - validate: Validation data (DataFrame)
    - test: Test data (DataFrame)
    - features: Features to use for clustering 
    - n_clusters: Number of clusters for KMeans (default is 4)

    Returns:
    - train: Training data with added 'cluster_1' column
    - validate: Validation data with added 'cluster_1' column
    - test: Test data with added 'cluster_1' column
    """
    # Train KMeans on training data
    X4 = train[features]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X4)

    # Add 'cluster_1' column to training data
    train['cluster_4'] = kmeans.predict(X4)

    # Add 'cluster_1' column to validation data
    V4 = validate[features]
    validate['cluster_4'] = kmeans.predict(V4)

    # Add 'cluster_1' column to test data
    T4 = test[features]
    test['cluster_4'] = kmeans.predict(T4)
    
    # visualize the clusters: 
    sns.scatterplot(x = 'pH', y = 'citric_acid', data = train, hue = 'cluster_4')
    plt.show()
    
    return train, validate, test


#--------------------------Modeling Functions--------------------#

def evaluate_reg(y, yhat):
    '''
    based on two series, y_act, y_pred, (y, yhat), we
    evaluate and return the root mean squared error
    as well as the explained variance for the data.
    
    returns: rmse (float), rmse (float)
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def evaluate_baseline(y_true):
    """
    Evaluate a baseline model using mean prediction.

    Parameters:
    - y_true: True target values

    Returns:
    - eval_df: DataFrame containing evaluation metrics for the baseline model
    """
    baseline = y_true.mean()
    baseline_array = np.repeat(baseline, y_true.shape[0])
    
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_array))
    baseline_r2 = r2_score(y_true, baseline_array)
    
    eval_df = pd.DataFrame([{
        'model': 'baseline',
        'rmse': baseline_rmse,
        'r2': baseline_r2
    }])
    
    return eval_df

def add_eval_df(model_name, rmse, r2, eval_df):
    """
    Add model evaluation metrics to an existing DataFrame.

    Parameters:
    - model_name: Name of the model
    - rmse: Root Mean Squared Error
    - r2: R-squared
    - eval_df: Existing DataFrame to which the model metrics will be added

    Returns:
    - eval_df: Updated DataFrame with the new model metrics
    """
    new_row = pd.DataFrame([{
        'model': model_name,
        'rmse': rmse,
        'r2': r2
    }])
    
    eval_df = pd.concat([eval_df, new_row], ignore_index=True)
    
    return eval_df

# create validation df 
def evaluate_validate(model_name, y_true, y_pred):
    """
    Evaluate a model's predictions and create a DataFrame with the results.

    Parameters:
    - model_name: Name of the model
    - y_true: True target values
    - y_pred: Predicted values from the model

    Returns:
    - model_df: DataFrame containing model evaluation metrics
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    val_df = pd.DataFrame([{
        'model': model_name,
        'val_rmse': rmse,
        'val_r2': r2
    }])
    
    return val_df
