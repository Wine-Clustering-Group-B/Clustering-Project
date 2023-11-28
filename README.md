# What Makes a High Quality Wine?
    Data consultation conducted for the California Wine Institute

by Nikki de Vries, Stephen Spivey, & Saul Gonzalez
|--------------------------------------------------------------------------------------------------------------|

<b>Project Description:  

This project contains the findings of research derived from the utilization of machine learning modeling combined with clustering to determine the highest
drivers of wine quality for the California Wine Institute.
    
|--------------------------------------------------------------------------------------------------------------|

<b>Project Goal:  Predict the quality of wine while incorporating unsupervised learning techniques.

|--------------------------------------------------------------------------------------------------------------|
<b>Initial Questions:

1. Are any of the features correlated? Can I apply some sort of feature selection?

2. Should I look at the top 'x' best and bottom 'x' worst wines for comparison? Is that a way to gain perspective?

3. What features would make good clusters? Should I use a heatmap?

4. Classification or regression? Should I do both for a comparison given the time I have to work on this?

5. Are all input variables relevant?

|--------------------------------------------------------------------------------------------------------------|

<b>Project Plan:

1. Create all the files needed to make a functioning project (.py and .ipynb files).

2. Create a .gitignore file and ignore the env.py file.

3. Start by acquiring data from 'Data.World Wine Quality Dataset' and document all my initial acquisition steps in the acquire.py file.

4. Using the prepare file, clean the data and split it into train, validatate, and test sets.

5. Explore the data. (Focus on the main main questions). Experiment with various feature combinations and clustering to gain insight, if some is found, to support hypothesis testing.

6. Answer all the questions with statistical testing.

7. Identify drivers of wine quality. Make prediction of wine quality using driving features of quality.

8. Document findings.

9. Add important finding to the final notebook.

10. Create csv file of test predictions on best performing model.

|--------------------------------------------------------------------------------------------------------------|

Data Dictionary:
These variables were based on physicochemical tests. Physicochemical tests are: tests that evaluate the materials of the container component or system to ensure purity and the absence of harmful contaminants or residuals from the manufacturing process.
|**Input Variables**|**Description**|
|----------|----------------|
|Fixed Acidity| Corresponds to the set of low volatility organic acids such as malic, lactic, tartaric or citric acids and is inherent to the characteristics of the sample.|
|Volatile Acidity | Corresponds to the set of short chain organic acids that can be extracted from the sample by means of a distillation process: formic acid, acetic acid, propionic acid and butyric acid. If there is a large amount of organic acids, it can lead to a unpleasant, vinegar taste. |
| Citric Acid | Often added to wines to increase acidity, which can result in a "fresh" flavor to the wine.|
| Residual Sugar | Measured in grams per liter, it is the natural graph sugar leftover in a wine after the alcholic fermentation finishes. For example, dry wine has 0-4 g/L, while sweet has 35 g/L. |
| Chlorides | Chloride ions give the perception of a 'salty' taste in the wine. |
| Free Sulfur Dioxide | Protects wine by scavenging oxygen and interrupting microbiological activity. |
| Total Sulfur Dioxide | The portion of SO2 that is free in the wine, plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugar. The levels are regulated by the U.S. Alcohol and Tobacco Tax and Trade Bureau (TTB). |
| Density | The measurement of how tightly a material is packed together. The density of wine is slightly less than that of water. |
| pH | Referred to as acidity or basicity. Wine has a lower pH (3.0-3.5) pH compared to water (7 pH). |
| Sulphates | Protects the wine against oxidation, which can effect the color and the taste of wine. |
| Alcohol |  Standard measure of how much alcohol (ethanol) there is within a given volume of the drink, in percentage terms. |
| Quality | Subtlety and complexity, aging potential, stylistic purity, varietal expression, ranking by experts, or consumer acceptance. Normally is on a scale of 1, being the worst, 10 being the best. |
| Wine color | The color of the wine, red or white. Red is annotated as '1' and White is annotated as '0'. |

|--------------------------------------------------------------------------------------------------------------|
<b>Conclusions:

<b>After acquiring & preparing the data, we conducted uni/bi/multi-variate exploration on the wine data to look at features and how they might impact the target 'quality'.

<b>We paired various features together and used clustering to observe potential relationships between the features.
     
<b>The results of our data exploration culminated in the resulting clusters and features being selected to go into regression modeling:

- Wine Color
- Chlorides
- Clustering of Alcohol and Sulphates
- Clustering of Citric Acid and pH
- Clustering of Free Sulfur Dioxide and Total Sulfur Dioxide
- Volatile Acidity and Density

<b>We chose to go with regression modeling due to all of our features being continuous. 

<b>We used the following regression models:
- Ordinary Least Squares (OLS)
- LassoLars
- Generalized Linear Model (GLM)

<b>We found that our Ordinary Least Squares model was the best performing model, showcasing a 12% average model prediction error on unseen data.
|--------------------------------------------------------------------------------------------------------------|

<b>Next Steps:

- We would look at conducting this entire study without the use of clustering, using the same models, to compare results and observe the impact of clustering to the modeling results.

- Furthermore, additional study on features for both red and white wines 'individually', given sufficient time, could prove insightful in determining the best drivers of quality for each colored wine.
    
- Lastly, if there happens to be additional data that becomes available, it could prove useful as there are likely other outside features that contribute to wine quality (grape quality, climate grapes grown, fermentation process, etc.) that could be stronger drivers of quality not provided by our current data source.
|--------------------------------------------------------------------------------------------------------------|

<b>Recommendations:  

- The data source showed a larger percentage of white wines produced compared to red wines, which could have produced a bias in the data that skewed the data. The data could be reduced to even out the differences between red and wines. 
    
- There could be an issue with oxidation in the wines. The lower quality wines have lower amounts of sulphates, and we think that by increasing the amount of sulphates, the oxidation issues would be remedied and improve the quality of wines. 
        
- Higher alcohol content is a major factor in the higher quality wines, specifically white wines. A two-fold effort can be enacted to maximize marketing towards white wine (where high quality is aplenty) and to chemically increase the alcohol while balancing the acidity to sufficiently improve quality in the red wines. 


<b>Steps to Reproduce Our Work:

1. Clone this repo.

2. Acquire the wine data from the Data.World Wine Quality Dataset.

3. Put the data in the file containing the cloned repo.

4. Run your notebook.