# What Makes a High Quality Wine?
    Data consultation conducted for the California Wine Institute

by Nikki de Vries, Stephen Spivey, & Saul Gonzalez
|------------------------------------------------------------------------------------------------------------------------------------------------------|

Project Description:  

This project contains the findings of research derived from the utilization of machine learning modeling combined with clustering to determine the highest
drivers of wine quality for the California Wine Institute.
    
|------------------------------------------------------------------------------------------------------------------------------------------------------|

Project Goal:  Predict the quality of wine while incorporating unsupervised learning techniques.

|------------------------------------------------------------------------------------------------------------------------------------------------------|

Project Plan:

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

|------------------------------------------------------------------------------------------------------------------------------------------------------|

Data Dictionary:
These variables were based on physciochemical tests. Physicochemical tests are: tests that evaluate the materials of the contrainer component or system to ensure purity and the absence of harmful contaminants or residuals from the manufacturing process.
|**Input Variables**|**Description**|
|----------|----------------|
|Fixed Acidity| corresponds to the set of low volatility organic acids such as malic, lactic, tartaric or citric acids and is inherent to the characteristics of the sample|
|Volatile Acidity | corresponds to the set of short chain organic acids that can be extracted from the sample by means of a distillation process: formic acid, acetic acid, propionic acid and butyric acid. If there is a large amount of organic acids it can lead to a upleasent, vingear taste. |
| Citric Acid | Often added to wines to increase acidity, which can result in a "fresh" flavor to the wine|
| Residual Sugar | Measured in grams per liter, it is the natural graph sugar leftover in a wine afte the alcholic fermentation finishes. For example dry wine has 0-4 g/L while sweet 35 g/L |
| Chlorides | chloride ions give the preception of a salty taste in the wine |
| Free Sulfur Dioxide | protects wine by scavenging oxygen and interrupting microbiological activity |
| Total Sulfur Dioxide | is the portion of SO2 that is free in the wine plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugar. The levels are regulated by the TTB |
| Density | is the measurement of how tightly a material is packed together. The density of wine is slightly less than that of water |
| ph | is referred to as acidity or basicitity. Wine has 3.0-3.5, water is 7 pH |
| Sulphates | Protects the wine agains oxidation, which can effect the color and the tatse of wine |
| Alcohol |  standard measure of how much alcohol (ethanol) there is within a given volume of the drink, in percentage terms |
| Quality | subtlety and complexity, aging potential, stylistic purity, varietal expression, ranking by experts, or consumer acceptance. Normally is on a scale of 1 being the worst, 10 being the best |
| Wine color | The color of the wine, red and white |
|------------------------------------------------------------------------------------------------------------------------------------------------------|

Conclusions:
|------------------------------------------------------------------------------------------------------------------------------------------------------|

Next Steps:

|------------------------------------------------------------------------------------------------------------------------------------------------------|

Recommendations:  

|------------------------------------------------------------------------------------------------------------------------------------------------------|

Steps to Reproduce Our Work:

1. Clone this repo.

2. Acquire the wine data from the Data.World Wine Quality Dataset.

3. Put the data in the file containing the cloned repo.

4. Run your notebook.