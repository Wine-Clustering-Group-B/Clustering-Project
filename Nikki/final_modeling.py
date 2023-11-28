# imports: 
 
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#visualizations
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.cluster import KMeans




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

