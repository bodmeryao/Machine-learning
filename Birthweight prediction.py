#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:57:23 2019

@author: bodmeryao
"""


# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.linear_model import LinearRegression
from sklearn.externals.six import StringIO # Saves an object in memory
from sklearn.tree import export_graphviz # Exports graphics
import pydotplus # Interprets dot objects
from IPython.display import Image # Displays an image on the frontend


# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file = 'birthweight-1.xlsx'

birthweight = pd.read_excel(file)

##############################################################################
#####                      Data understanding                            #####
##############################################################################

birthweight.shape

df_corr = birthweight.corr().round(2)

df_dropped = birthweight.dropna()
df_dropped_corr = df_dropped.corr().round(2)


plt.subplot(2, 2, 1)
sns.distplot(birthweight['monpre'],
             bins = 35,
             color = 'g')

plt.xlabel('Month Prenatal Care Began')


########################


plt.subplot(2, 2, 2)
sns.distplot(birthweight['npvis'],
             bins = 30,
             color = 'y')

plt.xlabel('Total Number of Prenatal Visits ')



########################


plt.subplot(2, 2, 3)
sns.distplot(birthweight['omaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('One Minute Apgar Score  ')



########################


plt.subplot(2, 2, 4)

sns.distplot(birthweight['fmaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Five Minute Apgar Score  ')



plt.tight_layout()
plt.savefig('Birthweight_2.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(birthweight['drink'],
             bins = 30,
             color = 'y')

plt.xlabel('avg drinks per week ')



########################

plt.subplot(2, 2, 2)
sns.distplot(birthweight['cigs'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('avg cigarettes per day ')



########################

plt.subplot(2, 2, 3)

sns.distplot(birthweight['bwght'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('birthweight, grams ')



########################

plt.subplot(2, 2, 4)
sns.distplot(birthweight['male'],
             bins = 35,
             color = 'g')

plt.xlabel('1 if baby male')



plt.tight_layout()
plt.savefig('Birthweight_3.png')

plt.show()

############################

# Understand the missing value

print(
      birthweight
      .isnull()
      .sum()
      )

sns.pairplot(birthweight)

sns.distplot(df_dropped['meduc'])

sns.distplot(df_dropped['feduc'])

sns.distplot(df_dropped['npvis'])

# Flagging missing values 
for col in birthweight:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birthweight[col].isnull().any():
        birthweight['m_'+col] = birthweight[col].isnull().astype(int)

# Creat model for predicting missing 'meduc' since no proper value is seen from 
# the others, as two most frequency values are equaly shown.

lm_full = smf.ols(formula = """meduc ~ df_dropped['feduc'] +
                                        df_dropped['mblck'] +
                                        df_dropped['fwhte'] + 
                                        df_dropped['foth']
                                           """,
                         data = df_dropped)

# Fitting Results
results = lm_full.fit()
print(results.summary())

# Tried to predict missing value of meduc by creating OLS model. However, the 
# R-squred is only 0.446, which means it won't be better than randomly impute. 

# As missing values are not too many, two 3s and a 7, drop all missing values

df_dropped = birthweight.dropna()

df_dropped_corr = df_dropped.corr().round(2)

print(df_dropped_corr)

df_dropped_corr.columns

# Use decision tree to give an idea of importance of variables

# Preparing data
dropped_data   = df_dropped.loc[:,['mage', 
                                'meduc', 
                                'monpre', 
                                'npvis', 
                                'fage', 
                                'feduc', 
                                'omaps', 
                                'fmaps',
                                'cigs', 
                                'drink', 
                                'male', 
                                'mwhte', 
                                'mblck', 
                                'moth', 
                                'fwhte', 
                                'fblck',
                                'foth']]

# Preparing the target variable
dropped_target = df_dropped.loc[:, 'bwght']

# Create train set and test set
X_train, X_test, y_train, y_test = train_test_split(
            dropped_data,
            dropped_target,
            test_size = 0.1,
            random_state = 508)

# Build the full tree
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))

# Defining a function to visualize feature importance
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_Feature_Importance.png')
        
########################
# Let's plot feature importance on the full tree.
plot_feature_importances(tree_full,
                         train = X_train,
                         export = False)

# From the graph, 'drink' as a variable is the most important one, after which,
# 'cigs', 'omaps', 'npivs', 'fage', and 'mage' are important as well. However, 'omaps' 
# represents the apgar score, which is a score system for after-birth baby. So
# that it should be denied.


##############################################################################
#####                  Model creation and testing                        #####
##############################################################################

##### OLS Medel #####
# As the dataset is tiny, with only 186 observations, it will not be a good idea
# spliting train/test datasets.

lm_full = smf.ols(formula = """bwght ~ df_dropped['drink'] +
                                        df_dropped['cigs'] +
                                        df_dropped['npvis'] +
                                        df_dropped['fage'] +
                                        df_dropped['mage'] +
                                        df_dropped['meduc'] +
                                        df_dropped['monpre'] +
                                        df_dropped['feduc'] +
                                        df_dropped['male'] +
                                        df_dropped['mwhte'] +
                                        df_dropped['mblck'] +
                                        df_dropped['moth'] +
                                        df_dropped['fwhte'] + 
                                        df_dropped['fblck'] +
                                        df_dropped['foth']
                                           """,
                         data = df_dropped)

# Fitting Results
results = lm_full.fit()
dir(results)

# Printing Summary Statistics
print(results.summary())

# Adjust variables based on p-values from last model, correlation table and importance.
lm_full = smf.ols(formula = """bwght ~ df_dropped['drink'] +
                                        df_dropped['cigs'] +
                                        df_dropped['mage'] +
                                        df_dropped['fage']
                                           """,
                         data = df_dropped)

# Fitting Results
results = lm_full.fit()

# Printing Summary Statistics
print(results.params)
print(results.summary())

# Checking predicted sale prices v. actual sale prices
predict = results.predict()
y_hat   = pd.DataFrame(predict).round(2)
resids  = results.resid.round(2)

# Plotting residuals
residual_analysis = pd.concat(
        [df_dropped.loc[:,'bwght'],
         y_hat,
         results.resid.round(2)],
         axis = 1)

sns.residplot(x = predict,
              y = df_dropped.loc[:,'bwght'])


plt.show()

print(f"""
Parameters:
{results.params.round(2)}

Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
# As is shown in the graph and the warning after running 
    
#####  Linear Regression  #####

# Preparing data
dropped_data   = df_dropped.loc[:,['mage',   
                                'cigs', 
                                'drink', 
                                'fage'
                                ]]

# Preparing the target variable
dropped_target = df_dropped.loc[:, 'bwght']

# Create train set and test set
X_train, X_test, y_train, y_test = train_test_split(
            dropped_data,
            dropped_target,
            test_size = 0.25,
            random_state = 508)

# Prepping the Model
lr = LinearRegression(fit_intercept = True)


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{lr_pred.round(2)}
""")


# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_ols_optimal)


# Let's compare the testing score to the training score.

print('The result of Linear Regression is:')
print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

# Plot the result
plt.scatter(y_test, lr_pred, color='blue', linewidth=3)
plt.plot(y_test, y_test, color='red')

plt.xticks(())
plt.yticks(())

plt.show()

##### Decision Tree  #####

# Preparing data
dropped_data   = df_dropped.loc[:,['mage',   
                                'cigs', 
                                'drink', 
                                'fage'
                                ]]

# Preparing the target variable
dropped_target = df_dropped.loc[:, 'bwght']

# Create train set and test set
X_train, X_test, y_train, y_test = train_test_split(
            dropped_data,
            dropped_target,
            test_size = 0.1,
            random_state = 508)

tree_leaf_5 = DecisionTreeRegressor(criterion = 'mse',
                                     max_depth = 5,
                                     random_state = 508)

tree_leaf_5.fit(X_train, y_train)

print('Result of Decision Tree is:')
print('Training Score', tree_leaf_5.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_5.score(X_test, y_test).round(4))

# Visualizing the tree
#dot_data = StringIO()

    
#export_graphviz(decision_tree = tree_leaf_5,
#                out_file = dot_data,
#                filled = True,
#                rounded = True,
#                special_characters = True)


#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


#Image(graph.create_png(),
#      height = 500,
#      width = 800)

#####  KNN analysis  ######

# Preparing data
dropped_data   = df_dropped.loc[:,['mage',   
                                'cigs', 
                                'drink', 
                                'fage'
                                ]]

# Preparing the target variable
dropped_target = df_dropped.loc[:, 'bwght']

# Create train set and test set
X_train, X_test, y_train, y_test = train_test_split(
            dropped_data,
            dropped_target,
            test_size = 0.1,
            random_state = 508)

# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 1)

# Checking the type of this new object
type(knn_reg)

# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)

# Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)

# Calling the score method, which compares the predicted values to the actual
# values
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []

# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)

for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

print(test_accuracy)

# The best results occur when k = 2.

# Building a model with k = 2
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 2)

# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)

# Scoring the model
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

print(f"""
The result of KNN model is {y_score.round(3)}.
""")
