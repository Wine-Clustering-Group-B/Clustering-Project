# imports: 
import numpy as np
import pandas as pd
import os 

# spliting data
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import MinMaxScaler
    
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
#imports for regression modeling

from sklearn.linear_model import LogisticRegression
# import K Nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier
# import Decision Trees:
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
# import Random Forest:
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

# interpreting our models:
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#feature selection
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn.linear_model import LinearRegression

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

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
#------------------------------------------------------------------------------
def citric_acid_pH_clusters(train, validate, test):
    X = train[['citric_acid', 'pH']]


    # MAKE the thing
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=4)
    
    # FIT the thing
    kmeans.fit(X)
    
    # USE (predict using) the thing 
    kmeans.predict(X)

    # make a new column names cluster in wines and X dataframe

    train['citric_pH_cluster'] = kmeans.predict(X)


    X['citric_pH_cluster'] = kmeans.predict(X)

    V1 = validate[['citric_acid', 'pH']]
    kmeans.predict(V1)
    
    T1 = test[['citric_acid', 'pH']]
    kmeans.predict(T1)
    
    # Cluster Centers aka centroids. -- The output is also not scaled; 
    # it would be scaled if the data used to fit was scaled.

    kmeans.cluster_centers_
    
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns[:2])
    
    train['citric_pH_cluster'] = train.citric_pH_cluster



    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'citric_acid', y = 'pH', data = train, hue = 'citric_pH_cluster')
    
    
    # plot cluster centers (centroids)
    centroids.plot.scatter(x = 'citric_acid', y = 'pH', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['citric_pH_cluster'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'citric_pH_cluster {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
def total_sulfur_dioxide_and_free_sulfur_dioxide_cluster(train, validate, test):
    X2 = train[['total_sulfur_dioxide', 'free_sulfur_dioxide']]

    # MAKE the thing
    from sklearn.cluster import KMeans

    kmeans2 = KMeans(n_clusters=4)
    
    # FIT the thing
    kmeans2.fit(X2)
    
    # USE (predict using) the thing 
    kmeans2.predict(X2)


    # make a new column names cluster in wines and X dataframe

    train['total_free_sulfur_dioxide_cluster'] = kmeans2.predict(X2)


    X2['total_free_sulfur_dioxide_cluster'] = kmeans2.predict(X2)

    V1 = validate[['total_sulfur_dioxide', 'free_sulfur_dioxide']]
    
    from sklearn.cluster import KMeans
    kmeans.predict(V1)
    
    T1 = test[['total_sulfur_dioxide', 'free_sulfur_dioxide']]
    kmeans.predict(T1)
    
    kmeans2.cluster_centers_
        
    centroids2 = pd.DataFrame(kmeans2.cluster_centers_, columns = X2.columns[:2])

    train['total_free_sulfur_dioxide_cluster'] = train.total_free_sulfur_dioxide_cluster


    
    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'total_sulfur_dioxide', y = 'free_sulfur_dioxide', data = train, hue = 'total_free_sulfur_dioxide_cluster')
    
    
    # plot cluster centers (centroids)
    centroids2.plot.scatter(x = 'total_sulfur_dioxide', y = 'free_sulfur_dioxide', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids2')
    
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['total_free_sulfur_dioxide_cluster'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'total_free_sulfur_dioxide_cluster {cluster}' for cluster in unique_clusters]
    
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
def sulphates_and_alcohol_clusters(train, validate, test):
    
    from sklearn.cluster import KMeans

    X3 = train[['sulphates', 'alcohol']]

    # MAKE the thing
    from sklearn.cluster import KMeans

    kmeans3 = KMeans(n_clusters=4)
    
    # FIT the thing
    kmeans3.fit(X3)
    
    # USE (predict using) the thing 
    kmeans3.predict(X3)
    
    # make a new column names cluster in wines and X dataframe

    train['sulphate_alcohol_cluster'] = kmeans3.predict(X3)


    X3['sulphate_alcohol_cluster'] = kmeans3.predict(X3)
    
    V1 = validate[['sulphates', 'alcohol']]
    kmeans.predict(V1)
    
    T1 = test[['sulphates', 'alcohol']]
    kmeans.predict(T1)    

    kmeans3.cluster_centers_
    
    centroids3 = pd.DataFrame(kmeans3.cluster_centers_, columns = X3.columns[:2])

    train['sulphate_alcohol_cluster'] = train.sulphate_alcohol_cluster


    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'sulphates', y = 'alcohol', data = train, hue = 'sulphate_alcohol_cluster')
    
    
    # plot cluster centers (centroids)
    centroids3.plot.scatter(x = 'sulphates', y = 'alcohol', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids3')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['sulphate_alcohol_cluster'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'sulphate_alcohol_cluster {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
def volatile_acidity_density_clusters(train, validate, test):
    
    from sklearn.cluster import KMeans

    X3 = train[['volatile_acidity', 'density']]

    # MAKE the thing
    from sklearn.cluster import KMeans

    kmeans3 = KMeans(n_clusters=4)
    
    # FIT the thing
    kmeans3.fit(X3)
    
    # USE (predict using) the thing 
    kmeans3.predict(X3)
    
    # make a new column names cluster in wines and X dataframe

    train['volatile_acidity_density_cluster'] = kmeans3.predict(X3)


    X3['volatile_acidity_density_cluster'] = kmeans3.predict(X3)
    
    V1 = validate[['volatile_acidity', 'density']]
    kmeans.predict(V1)
    
    T1 = test[['volatile_acidity', 'density']]
    kmeans.predict(T1)    
    

    kmeans3.cluster_centers_
    
    centroids3 = pd.DataFrame(kmeans3.cluster_centers_, columns = X3.columns[:2])

    train['volatile_acidity_density_cluster'] = train.volatile_acidity_density_cluster


    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'volatile_acidity', y = 'density', data = train, hue = 'volatile_acidity_density_cluster')
    
    
    # plot cluster centers (centroids)
    centroids3.plot.scatter(x = 'volatile_acidity', y = 'density', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids3')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['volatile_acidity_density_cluster'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'volatile_acidity_density_cluster {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
def get_baseline(train):
    X = train.drop(columns={'quality'})
    y = train['quality']
    
    from sklearn.linear_model import LinearRegression

    #make
    lm = LinearRegression()
    #fit
    lm.fit(X,y)
    #use
    yhat = lm.predict(X)
    
    baseline_med = y.median()
    baseline_mean = y.mean()
    
    y_pred = pd.DataFrame(
    {
    'y_act': y.values,
    'yhat' : yhat,
    'baseline_med': baseline_med,
    'baseline_mean': baseline_mean
    }, index=train.index)
    
    # compute the error on these two baselines: I want the lower RSME for the baseline
    mean_baseline_rmse = mean_squared_error(y_pred.baseline_mean, y) ** (1/2)
    med_baseline_rmse = mean_squared_error(y_pred.baseline_med, y) ** (1/2)
    
    baseline = mean_baseline_rmse
    baseline_rmse = mean_baseline_rmse
    
    return baseline, baseline_rmse
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def OLS(X_train, y_train, baseline, X_val, y_val):
    from sklearn.linear_model import LinearRegression
    # MAKE THE THING: create the model object
    linear_model = LinearRegression()
    #1. FIT THE THING: fit the model to training data
    OLSmodel = linear_model.fit(X_train, y_train)

    #2. USE THE THING: make a prediction
    y_train_pred = linear_model.predict(X_train)
    #3. Evaluate: RMSE
    rmse_train = mean_squared_error(y_train, y_train_pred) ** (.5) # 0.5 to get the root
    
    # convert results into dataframe
    result = pd.DataFrame({
        "target": y_train,
        "OLS_prediction": y_train_pred,
        "baseline_pred": baseline
    })
    
    # convert to dataframe
    X_val = pd.DataFrame(X_val)
    X_val[X_val.columns] = X_val
    X_val = X_val[X_val.columns]
    
    #2. USE THE THING: make a prediction
    y_val_pred = linear_model.predict(X_val)
    
    #3. Evaluate: RMSE
    rmse_val = mean_squared_error(y_val, y_val_pred) ** (.5) # 0.5 to get the root
    
    OLSmodel.coef_
    
    
    print(f"OLS Regressor \nRMSE_train {rmse_train} \
\nRMSE_validate {rmse_val} \nR2_validate {explained_variance_score(y_val, y_val_pred)}")
    
#------------------------------------------------------------------------------
def LassoLars(X_train, y_train, baseline, X_val, y_val):
    from sklearn.linear_model import LassoLars

    # MAKE THE THING: create the model object
    lars = LassoLars(alpha= 1.0)
    
    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    laslars = lars.fit(X_train, y_train)
    
    #2. USE THE THING: make a prediction
    y_train_pred_lars = lars.predict(X_train)
    
    #3. Evaluate: RMSE
    rmse_train_lars = mean_squared_error(y_train, y_train_pred_lars) ** (0.5)
    
    # predict validate
    y_val_pred_lars = lars.predict(X_val)
    
    # evaluate: RMSE
    rmse_val_lars = mean_squared_error(y_val, y_val_pred_lars) ** (0.5)
    
    # how important is each feature to the target
    laslars.coef_
    
    print(f"""RMSE for Lasso + Lars
_____________________
Training/In-Sample: {rmse_train_lars}, 
Validation/Out-of-Sample:  {rmse_val_lars}
Difference:  {rmse_val_lars - rmse_train_lars}""")
#------------------------------------------------------------------------------

def GLM(X_train, y_train, baseline, X_val, y_val):
    from sklearn.linear_model import TweedieRegressor
    # MAKE THE THING: create the model object
    glm_model = TweedieRegressor(alpha= 1.0, power= 1)
    
    #1. FIT THE THING: fit the model to training data
    # We must specify the column in y_train, since we have converted it to a dataframe from a series!
    tweedieReg = glm_model.fit(X_train, y_train)
    
    #2. USE THE THING: make a prediction
    y_train_pred_tweedie = glm_model.predict(X_train)
    
    #3. Evaluate: RMSE
    rmse_train_tweedie = mean_squared_error(y_train, y_train_pred_tweedie) ** (0.5)
    
    # predict validate
    y_val_pred_tweedie = glm_model.predict(X_val)
    
    # evaluate: RMSE
    rmse_val_tweedie = mean_squared_error(y_val, y_val_pred_tweedie) ** (0.5)
    
    # how important is each feature to the target
    tweedieReg.coef_
    
    print(f"""RMSE for Lasso + Lars
_____________________
Training/In-Sample: {rmse_train_tweedie}, 
Validation/Out-of-Sample:  {rmse_val_tweedie}
Difference:  {rmse_val_tweedie - rmse_train_tweedie}""")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def break_em_out(preprocessed_train, preprocessed_validate, preprocessed_test):
    # train set
    X_train = preprocessed_train.drop(columns=['quality']) 
    y_train = preprocessed_train['quality']
    
    # validate set
    X_val = preprocessed_validate.drop(columns=['quality']) 
    y_val = preprocessed_validate['quality']
    
    # test
    X_test = preprocessed_test.drop(columns=['quality'])
    y_test = preprocessed_test['quality']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

#------------------------------------------------------------------------------
