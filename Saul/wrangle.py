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




#------------------------------------------------------------------------------
def get_connection_url(db, user=env.user, host=env.host, password=env.password):
    """
    This function will:
    - take username, pswd, host credentials from imported env module
    - output a formatted connection_url to access mySQL db
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

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
    train['Wine_Color'] = train.Wine_Color.map({'Red Wine': 1, 'White Wine': 0})
    validate['Wine_Color'] = validate.Wine_Color.map({'Red Wine': 1, 'White Wine': 0})
    test['Wine_Color'] = test.Wine_Color.map({'Red Wine': 1, 'White Wine': 0})
    return train, validate, test
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
def acquire_data():
    white = pd.read_csv('white_wine.csv')
    white_wines = 'White Wine'
    white['Wine_Color'] = white_wines
    red = pd.read_csv('red_wine.csv')
    red_wines = 'Red Wine'
    red['Wine_Color'] = red_wines
    frames = [white, red]
    wines = pd.concat(frames)
    wines = wines.rename(columns=({'fixed acidity': 'fixed_acidity', 'volatile acidity': 'volatile_acidity', 'citric acid': 'citric_acid', 'residual sugar': 'residual_sugar', 'free sulfur dioxide': 'free_sulfur_dioxide', 'total sulfur dioxide':'total_sulfur_dioxide'}))
    
    return wines

#------------------------------------------------------------------------------
def heatmap(train):
    train_corr = train.corr()

# Pass my correlation matrix to Seaborn's heatmap.
    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
              'linecolor':'k','rasterized':False, 'edgecolor':'w', 
              'capstyle':'projecting',}
    plt.figure(figsize=(20,10))
    sns.heatmap(train_corr, cmap='Purples', annot=True, mask= np.triu(train_corr), **kwargs)
    #plt.ylim(10, 10)
    
    plt.show()
#------------------------------------------------------------------------------
def cluster1(train):
    X = train[['citric_acid', 'pH']]

    # MAKE the thing
    kmeans = KMeans(n_clusters=3)
    
    # FIT the thing
    kmeans.fit(X)
    
    # USE (predict using) the thing 
    kmeans.predict(X)
    
    # make a new column names cluster in wines and X dataframe

    train['cluster'] = kmeans.predict(X)
    
    X['cluster'] = kmeans.predict(X)
    
    # Cluster Centers aka centroids. -- The output is also not scaled; 
    # it would be scaled if the data used to fit was scaled.

    kmeans.cluster_centers_
    
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns[:2])
    
    train['cluster'] = train.cluster
    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'citric_acid', y = 'pH', data = train, hue = 'cluster')
    
    
    # plot cluster centers (centroids)
    centroids.plot.scatter(x = 'citric_acid', y = 'pH', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['cluster'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'Cluster {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');


#------------------------------------------------------------------------------
def cluster2(train):
    X2 = train[['sulphates', 'free_sulfur_dioxide']]

    # MAKE the thing
    kmeans2 = KMeans(n_clusters=3)
    
    # FIT the thing
    kmeans2.fit(X2)
    
    # USE (predict using) the thing 
    kmeans2.predict(X2)
    
    # make a new column names cluster in wines and X dataframe

    train['cluster2'] = kmeans2.predict(X2)
    
    X2['cluster2'] = kmeans2.predict(X2)
    
    kmeans2.cluster_centers_
        
    centroids2 = pd.DataFrame(kmeans2.cluster_centers_, columns = X2.columns[:2])

    train['cluster2'] = train.cluster2

    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'sulphates', y = 'free_sulfur_dioxide', data = train, hue = 'cluster2')
    
    
    # plot cluster centers (centroids)
    centroids2.plot.scatter(x = 'sulphates', y = 'free_sulfur_dioxide', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids2')
    
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['cluster2'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'Cluster2 {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
    
#------------------------------------------------------------------------------
def cluster3(train):
    
    from sklearn.cluster import KMeans

    X3 = train[['free_sulfur_dioxide', 'alcohol']]

    # MAKE the thing
    kmeans3 = KMeans(n_clusters=3)
    
    # FIT the thing
    kmeans3.fit(X3)
    
    # USE (predict using) the thing 
    kmeans3.predict(X3)
    
    # make a new column names cluster in wines and X dataframe

    train['cluster3'] = kmeans3.predict(X3)

    X3['cluster3'] = kmeans3.predict(X3)
    
    kmeans3.cluster_centers_
    
    centroids3 = pd.DataFrame(kmeans3.cluster_centers_, columns = X3.columns[:2])

    train['cluster3'] = train.cluster3

    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'free_sulfur_dioxide', y = 'alcohol', data = train, hue = 'cluster3')
    
    
    # plot cluster centers (centroids)
    centroids3.plot.scatter(x = 'free_sulfur_dioxide', y = 'alcohol', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids3')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['cluster3'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'Cluster3 {cluster}' for cluster in unique_clusters]
    
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
def scale_the_data(train, validate, test):
    from sklearn.preprocessing import MinMaxScaler

    continuous_features = ['fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH', 'alcohol'] # The values in these columns need to be scaled
    # make an object:
    scaler = MinMaxScaler()

    # fit the thing
    # if you are using an sklearn object, make sure you only call fit on train!
    scaler.fit(train[continuous_features])
    
    # Use the thing
    train[['fixed_acidity_scaled', 'residual_sugar_scaled', 'free_sulfur_dioxide_scaled', 'total_sulfur_dioxide_scaled', 'pH_scaled', 'alcohol_scaled']] = scaler.transform(train[continuous_features])
    preprocessed_train = train.drop(columns=['fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH', 'alcohol'])

    # Use the thing on Validate and Test
    validate[['fixed_acidity_scaled', 'residual_sugar_scaled', 'free_sulfur_dioxide_scaled', 'total_sulfur_dioxide_scaled', 'pH_scaled', 'alcohol_scaled']] = scaler.transform(validate[continuous_features])
    test[['fixed_acidity_scaled', 'residual_sugar_scaled', 'free_sulfur_dioxide_scaled', 'total_sulfur_dioxide_scaled', 'pH_scaled', 'alcohol_scaled']] = scaler.transform(test[continuous_features])
    
    #### Now to drop the columns that are not needed for the Machine Learning algorithms
    preprocessed_validate = validate.drop(columns=['fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH', 'alcohol'])
    preprocessed_test = test.drop(columns=['fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH', 'alcohol'])
    
    return preprocessed_train, preprocessed_validate, preprocessed_test


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
    X_val = X_val[X_train.columns]
    
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
def PolyFeat(X_train, y_train, baseline, X_val, y_val, X_test, y_test):
    from sklearn.preprocessing import PolynomialFeatures

    # convert to dataframe
    X_test = pd.DataFrame(X_test)
    X_test[X_test.columns] = X_test
    X_test = X_test[X_train.columns]
    
    #1. Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2) #Quadratic aka x-squared
    
    #1. Fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    
    #1. Transform X_validate_scaled & X_test_scaled 
    X_val_degree2 = pf.transform(X_val)
    X_test_degree2 = pf.transform(X_test)
    
    from sklearn.linear_model import LinearRegression

    #2.1 MAKE THE THING: create the model object
    poly_model = LinearRegression()
    
    
    #2.2 FIT THE THING: fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    from sklearn.linear_model import LinearRegression

    polyFeat = linear_model.fit(X_train_degree2, y_train)
    
    #3. USE THE THING: predict train
    from sklearn.linear_model import LinearRegression

    y_train_pred_poly = linear_model.predict(X_train_degree2)
    
    #4. Evaluate: rmse
    rmse_train_poly = mean_squared_error(y_train, y_train_pred_poly) ** (0.5)
    
    from sklearn.linear_model import LinearRegression

    # predict validate
    y_val_pred_poly = linear_model.predict(X_val_degree2)
    
    # evaluate: RMSE
    rmse_val_poly = mean_squared_error(y_val, y_val_pred_poly) ** (0.5)
    
    # how important is each feature to the target
    polyFeat.coef_
    
    
    print(f"""RMSE for Polynomial Model, degrees=2
_____________________________________________
Training/In-Sample:  {rmse_train_poly} 
Validation/Out-of-Sample:  {rmse_val_poly}
difference:  {rmse_val_poly - rmse_train_poly}""")
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
def OLS_test(X_train, y_train, baseline, X_val, y_val, X_test, y_test):
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
    X_val = X_val[X_train.columns]
    
    #2. USE THE THING: make a prediction
    y_val_pred = linear_model.predict(X_val)
    
    #3. Evaluate: RMSE
    rmse_val = mean_squared_error(y_val, y_val_pred) ** (.5) # 0.5 to get the root
    
    # convert to dataframe
    X_test = pd.DataFrame(X_test)
    X_test[X_test.columns] = X_test
    X_test = X_test[X_train.columns]
    
    #2. USE THE THING: make a prediction
    y_test_pred = linear_model.predict(X_test)
    
    #3. Evaluate: RMSE
    rmse_test = mean_squared_error(y_test, y_test_pred) ** (.5) # 0.5 to get the root
    
    OLSmodel.coef_

    
    print(f"""RMSE for Ordinary Least Squares Test Model
_____________________________________________
Baseline: {baseline}
Training/In-Sample:  {rmse_train} 
Validation/Out-of-Sample:  {rmse_val}
Test/Out-of-Sample: {rmse_test}
difference:  {rmse_test - rmse_val}""")
#------------------------------------------------------------------------------

    
def citric_acid_pH_clusters(train):
    X = train[['citric_acid', 'pH']]

    # MAKE the thing
    kmeans = KMeans(n_clusters=6)
    
    # FIT the thing
    kmeans.fit(X)
    
    # USE (predict using) the thing 
    kmeans.predict(X)
    
    # make a new column names cluster in wines and X dataframe

    train['cluster'] = kmeans.predict(X)
    
    X['cluster'] = kmeans.predict(X)
    
    # Cluster Centers aka centroids. -- The output is also not scaled; 
    # it would be scaled if the data used to fit was scaled.

    kmeans.cluster_centers_
    
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns = X.columns[:2])
    
    train['cluster'] = train.cluster
    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'citric_acid', y = 'pH', data = train, hue = 'cluster')
    
    
    # plot cluster centers (centroids)
    centroids.plot.scatter(x = 'citric_acid', y = 'pH', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['cluster'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'Cluster {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
def total_sulfur_dioxide_and_free_sulfur_dioxide_cluster(train):
    X2 = train[['total_sulfur_dioxide', 'free_sulfur_dioxide']]

    # MAKE the thing
    kmeans2 = KMeans(n_clusters=4)
    
    # FIT the thing
    kmeans2.fit(X2)
    
    # USE (predict using) the thing 
    kmeans2.predict(X2)
    
    # make a new column names cluster in wines and X dataframe

    train['cluster2'] = kmeans2.predict(X2)
    
    X2['cluster2'] = kmeans2.predict(X2)
    
    kmeans2.cluster_centers_
        
    centroids2 = pd.DataFrame(kmeans2.cluster_centers_, columns = X2.columns[:2])

    train['cluster2'] = train.cluster2

    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'total_sulfur_dioxide', y = 'free_sulfur_dioxide', data = train, hue = 'cluster2')
    
    
    # plot cluster centers (centroids)
    centroids2.plot.scatter(x = 'total_sulfur_dioxide', y = 'free_sulfur_dioxide', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids2')
    
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['cluster2'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'Cluster2 {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
def sulphates_and_alcohol_clusters(train):
    
    from sklearn.cluster import KMeans

    X3 = train[['sulphates', 'alcohol']]

    # MAKE the thing
    kmeans3 = KMeans(n_clusters=5)
    
    # FIT the thing
    kmeans3.fit(X3)
    
    # USE (predict using) the thing 
    kmeans3.predict(X3)
    
    # make a new column names cluster in wines and X dataframe

    train['cluster3'] = kmeans3.predict(X3)

    X3['cluster3'] = kmeans3.predict(X3)
    
    kmeans3.cluster_centers_
    
    centroids3 = pd.DataFrame(kmeans3.cluster_centers_, columns = X3.columns[:2])

    train['cluster3'] = train.cluster3

    # lets visualize the clusters along with the centers on unscaled data
    plt.figure(figsize=(14, 9))
    plt.figure(figsize=(14, 9))
    
    
    # scatter plot of data with hue for cluster
    sns.scatterplot(x = 'sulphates', y = 'alcohol', data = train, hue = 'cluster3')
    
    
    # plot cluster centers (centroids)
    centroids3.plot.scatter(x = 'sulphates', y = 'alcohol', ax = plt.gca(), color ='k', alpha = 0.3, s = 800, marker = (8,1,0), label = 'centroids3')
    
    plt.title('Visualizing Cluster Centers')
    
    # Get unique cluster labels
    unique_clusters = train['cluster3'].unique()
    
    # Create legend labels for clusters
    cluster_labels = [f'Cluster3 {cluster}' for cluster in unique_clusters]
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left');
#------------------------------------------------------------------------------
