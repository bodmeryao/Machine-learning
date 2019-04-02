#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:49:11 2019

@author: bodmeryao
"""

# Loading libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsClassifier # KNN for Classifier
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier # Classification trees
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects
from sklearn.model_selection import RandomizedSearchCV


# Load dataset
file = 'GOT_character_predictions.xlsx'


data_df = pd.read_excel(file)


###############################################################################
#####                 Fundamental Dataset Exploration                     #####
###############################################################################

# Column names
data_df.columns


# Displaying the first rows of the DataFrame
print(data_df.head())


# Dimensions of the DataFrame
data_df.shape


# Information about each variable
data_df.info()


# Descriptive statistics
data_df.describe().round(2)


# Plot the target variable
sns.distplot(data_df['isAlive'])


# Learn about each column
for col in data_df:
    print(col, data_df[col].dropna().value_counts())


# Missing value understanding
print(
      data_df
      .isnull()
      .sum()
      )


# Create a column of missing value of a charactor, named after 'misvalue'
# I assume if a charactor has a lot of missing values, the charactor may not be
# a key person that worth the author talking more about. By isolate such a column,
# it may separate all charators into different levels of importance in the book.

for row in data_df:
    data_df['misvalue'] =data_df.shape[1] - data_df.count(axis=1)


##### Imputing missing values #####
    
### Title ###
    
fill = 'Civilian'


data_df['title']=data_df['title'].fillna(fill)


type(data_df['title'])


data_df['title'].dropna().value_counts()


# Grouping different titles
# By doing external research, convert similar titles into one title

for val in enumerate(data_df['title']):
    
    if 'Hand of the King' in val[1]:
        data_df.loc[val[0],'title']='Ser'
        
    if 'Septa' in val[1]:
        data_df.loc[val[0],'title']='Seven Kingdoms'
        
    if 'King' in val[1]:
        data_df.loc[val[0],'title']='King'
        
    if 'Lord' in val[1]:
        data_df.loc[val[0],'title']='King'
        
    if 'Prince' in val[1]:
        data_df.loc[val[0],'title']='Prince'
        
    if 'Lady' in val[1]:
        data_df.loc[val[0],'title']='Lady'
        
    if 'Maester' in val[1]:
        data_df.loc[val[0],'title']='Maester'
        
    if 'Archmaester' in val[1]:
        data_df.loc[val[0],'title']='Maester'
        
    if 'Bloodrider' in val[1]:
        data_df.loc[val[0],'title']='Khal'
        
    if 'Knight' in val[1]:
        data_df.loc[val[0],'title']='Ser'
        
    if 'Khal' in val[1]:
        data_df.loc[val[0],'title']='Khal' 
        
    if 'Good Master' in val[1]:
        data_df.loc[val[0],'title']='Maester'
    

# Identify those titles appears less than 8 times as 'Others'

# Create a tc series include all titles that appears less than 8 times        

tc=data_df.loc[:,'title'].dropna().value_counts()


tc=tc[10:]


# Convert index with titles' name into list 'juc'

juc=tc.index.tolist()


# Replace all not that frequent titles into 'Others'

for val in enumerate(data_df['title']):
    if val[1] in juc:
        data_df.loc[val[0],'title']='Other'


# Dummilize the dataset

titled = pd.get_dummies(list(data_df['title']), drop_first=True) 


### house ###
# As checking the dataset, one thing that worth mentioned that is for Targaryen
# family, they don't have a house under their names, which should be Valyria.

for val in enumerate(data_df['name']):
    if 'Targaryen' in val[1]:
        data_df.loc[val[0], 'house'] = 'House Targaryen'
    if 'Euron' in val[1]:
        data_df.loc[val[0],'house']= 'Sunderly'
    if 'Stark' in val[1]:
        data_df.loc[val[0],'house'] = 'House Stark'
    if 'Baratheon' in val[1]:
        data_df.loc[val[0],'house'] = 'Great House'
    

data_df['house'].dropna().value_counts()


fill='None'


data_df['house']=data_df['house'].fillna(fill)


# Identify those Houses appears less than 10 times as 'Others'

# Create a hc series include all houses that appears less than 10 times        

hc=data_df.loc[:,'house'].dropna().value_counts()


hc=hc[25:]


# Convert index with titles' name into list 'juc'

huc=hc.index.tolist()


# Replace all not that frequent titles into 'Others'

for val in enumerate(data_df['house']):
    if val[1] in huc:
        data_df.loc[val[0],'house']='Other'
        
    
# Dummilize the dataset

housed = pd.get_dummies(list(data_df['house']), drop_first=True) 

        
### Culture ###

# As doing external research, a lot of missing values are actually have information
# from books. However, more than 2/3 of the whole dataset are missing, which means
# it is impossible to impute all missing values manually.

data_df['culture'].dropna().value_counts()


### TEST by imputing 'Unknown' to missing values ###
# Assume that culture would not affect the charactor's possibility of survival.

fill='Unknown'

data_df['culture']=data_df['culture'].fillna(fill)


for val in enumerate(data_df['culture']):
    if 'Dorn' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Donish'
    if 'Iron' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Ironborn'
    if 'iron' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Ironborn'
    if 'Andal' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Andals'
    if 'Free' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Free Folk'
    if 'Wester' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Westermen'
    if 'free' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Free Folk'
    if 'north' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Northmen'
    if 'Northern' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Northmen'
    if 'Meereen' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Meereenese'
    if 'River' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Rivermen'
    if 'wester' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Westermen'
    if 'Vale' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Valemen'
    if 'Storm' in val[1]:
        data_df.loc[val[0], 'culture'] = 'Stormlander'
    

# Identify those cultures appears less than 8 times as 'Others'

# Create a cc series include all culture that appears less than 8 times        

cc=data_df.loc[:,'culture'].dropna().value_counts()


cc=cc[14:]


# Convert index with cultures' name into list 'juc'

cuc=cc.index.tolist()


# Replace all not that frequent cultures into 'Others'

for val in enumerate(data_df['culture']):
    if val[1] in cuc:
        data_df.loc[val[0],'culture']='Other'


# Dummilize the dataset

cultured = pd.get_dummies(list(data_df['culture']), drop_first=True) 
        

### dateOfBirth & age ###

# Impute 0 represent unknown, since most of the charactors who do not have age 
# information are not important ones, that no way to get these information by 
# external research.

data_df['dateOfBirth'].dropna().value_counts()


data_df['age'].dropna().value_counts()


fill=0


data_df['dateOfBirth']=data_df['dateOfBirth'].fillna(fill)


fill=pd.np.mean(data_df['age'])


data_df['age']=data_df['age'].fillna(fill)


# Create columns of dead year, with assumption that during the year has a war, 
# more people would be died as a result.

for row in data_df:
    data_df['dueyear'] = data_df['dateOfBirth']+data_df['age']


data_df['dueyear']=data_df['dueyear'].fillna(fill)


# Group age data into 6 differet groups: old'>70', mold'70>x>40', m'40>x>20'
# y'20>x>5', b'<5' and n for unknown

for val in enumerate(data_df['age']):
    if val[1]>70:
        data_df.loc[val[0], 'age']='old'
    elif val[1]>40:
        data_df.loc[val[0], 'age']='mid-old'
    elif val[1]>20:
        data_df.loc[val[0], 'age']='middle age'
    elif val[1]>5:
        data_df.loc[val[0], 'age']='young'
    elif val[1]!=0:
        data_df.loc[val[0], 'age']='baby'
    else:
        data_df.loc[val[0], 'age']='unknown'

# Group dateOfBirth data into 7 different groups, which are AC101，AC171, AC209,
# AC233, AC276, and AC298, which are decided based on external research.
        
data_df['dateOfBirth'].drop_duplicates().sort_values()


for val in enumerate(data_df['dateOfBirth']):
    if val[1]>298:
        data_df.loc[val[0], 'dateOfBirth']='AC298'
    elif val[1]>276:
        data_df.loc[val[0], 'dateOfBirth']='AC276'
    elif val[1]>233:
        data_df.loc[val[0], 'dateOfBirth']='AC233'
    elif val[1]>209:
        data_df.loc[val[0], 'dateOfBirth']='AC209'
    elif val[1]>171:
        data_df.loc[val[0], 'dateOfBirth']='AC171'
    elif val[1]>101:
        data_df.loc[val[0], 'dateOfBirth']='AC101'
    elif val[1]!=0:
        data_df.loc[val[0], 'dateOfBirth']='BEFORE'
    else:
        data_df.loc[val[0], 'dateOfBirth']='UnKnown'


# Dummilize the dataset

aged = pd.get_dummies(list(data_df['age']), drop_first=True) 


birthd = pd.get_dummies(list(data_df['dateOfBirth']), drop_first=True) 


### Spouse ###
# Even missing values of Spouse are not as many as 'mother', it is still roughly
# 90% of values are missing. Furthermore, the variable 'isAliveSpouse' can give
# enough informtion about this topic.

data_df=data_df.drop(['spouse'], axis=1)

# Books charactors appear
# Though no missing values are shown in these 5 variables, there is an assumption
# that charactors shown in more books are more valuable than those appeared in 
# only one book.
# Create a column 'Bookscount'

data_df.rename(columns={'book1_A_Game_Of_Thrones':'Book1',
                        'book2_A_Clash_Of_Kings':'Book2',
                        'book3_A_Storm_Of_Swords':'Book3',
                        'book4_A_Feast_For_Crows':'Book4',
                        'book5_A_Dance_with_Dragons':'Book5'}, inplace=True)
    
    
data_df['Bookscount']=data_df['Book1']+data_df['Book2']+data_df['Book3']+data_df['Book4']+data_df['Book5']

### isAliveSpouse & isMarried ###
# Combine the result from column 'isMarried', impute 0 in 'isAliveSpouse', representing
# a dead spouse. Otherwise, impute 2, representing no spouse.

fill=2


data_df['isAliveSpouse']=data_df['isAliveSpouse'].fillna(fill)


### isAliveMother & isAliveFather & isAliveHeir ###
### TEST - Keep these three columns ###
# Impute 2 to all missing values, repesenting no mother/father/heir information

data_df['isAliveMother']=data_df['isAliveMother'].fillna(fill)
    

data_df['isAliveFather']=data_df['isAliveFather'].fillna(fill)


data_df['isAliveHeir']=data_df['isAliveHeir'].fillna(fill)


### 'mother', 'father' and 'heir' ### 
# There are too many missing values, only roughly 1% of whom have data. So, 
# these variables are dropped.

data_df=data_df.drop(['mother','father','heir'], axis=1)


### name ###
# Extract first name and family name from name column

name=data_df['name'].str.split(' ')


for i in range(len(name)):
    data_df.loc[i,'firstname']=name[i][0]
    if len(name[i])==2:
        data_df.loc[i,'familyname']=name[i][1]

fill='None'

data_df['familyname']=data_df['familyname'].fillna(fill)


# Identify those familynames appears less than 10 times as 'Others'

# Create a fnc series include all familynames that appears less than 10 times        

fnc=data_df.loc[:,'familyname'].dropna().value_counts()


fnc=fnc[25:]


# Convert index with titles' name into list 'juc'

fnuc=hc.index.tolist()


# Replace all not that frequent titles into 'Others'

for val in enumerate(data_df['familyname']):
    if val[1] in fnc:
        data_df.loc[val[0],'familyname']='Other'


# Dummilize the dataset

fnd = pd.get_dummies(list(data_df['familyname']), drop_first=True) 


# Further Dataset cleaness

# Convert variables to integer

data_df['isAliveMother']=data_df['isAliveMother'].astype('int')

data_df['isAliveFather']=data_df['isAliveFather'].astype('int')

data_df['isAliveHeir']=data_df['isAliveHeir'].astype('int')

data_df['isAliveSpouse']=data_df['isAliveSpouse'].astype('int')

data_df['popularity']=data_df['popularity'].astype('int')

data_df['dueyear']=data_df['dueyear'].astype('int')

    
###############################################################################
#####                   Model Creation and Testing                        #####
###############################################################################

##### Data preparation #####

# Concatenating the dataset and dummies

data_df.shape


data_df=pd.concat([data_df.iloc[:,:],
                   aged,
                   #birthd,
                   #cultured,
                   #fnd,
                   housed,
                   titled
                   ], axis = 1)
    
    
# Create dataset used for model prediction

datagt = data_df.drop(['S.No',
                       'name',
                       'isAlive',
                       'title',
                       'house',
                       'firstname',
                       'familyname',
                       'culture',
                       'dateOfBirth',
                       'age',
                       'isAliveMother',
                       'isAliveFather',
                       'isAliveSpouse',
                       'isMarried',
                       'popularity'], axis = 1)

    
datagt_target = data_df.loc[:, 'isAlive']


# Train/Test data split

X_train, X_test, y_train, y_test = train_test_split(
            datagt,
            datagt_target,
            test_size = 0.1,
            random_state = 508,
            stratify = datagt_target)


# Merge X-train and Y-train dataset

datagt_train = pd.concat([X_train, y_train], axis = 1)


####################################
##### KNN classification model #####
####################################

# Running the neighbor optimization code with a small adjustment for classification

training_accuracy = []


test_accuracy = []


neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


#fig, ax = plt.subplots(figsize=(12,9))
#plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
#plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("n_neighbors")
#plt.legend()
#plt.show()


# Looking for the highest test accuracy
#print(test_accuracy)


# Printing highest test accuracy
#print(test_accuracy.index(max(test_accuracy)) + 1)


# Parameter tuning with GridSearchCV

# Creating a hyperparameter grid

n_neighbors = pd.np.arange(1, 51, 1)
weights = ['uniform', 'distance']
algorithm = ['auto','ball_tree','kd_tree','brute']


param_grid = {'n_neighbors' : n_neighbors,
              'weights' : weights,
              'algorithm' : algorithm}

# Create a KNN model

clf = KNeighborsClassifier()


clf_cv = GridSearchCV(clf, param_grid, n_jobs = -1, cv = 3)

# Fit it to the training data

clf_cv.fit(X_train, y_train)



# Print the optimal parameters and best score

#print("Tuned Parameter:", clf_cv.best_params_)
#print("Tuned Accuracy:", clf_cv.best_score_.round(4))


# Test on 14 neighbors, coming from best n_neighbors test result
knn_clf = KNeighborsClassifier(n_neighbors = 14)


# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)


# Let's compare the testing score to the training score.
#print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
#print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# Generating Predictions based on the optimal KNN model
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)

#print(confusion_matrix(y_true = y_test,
#                       y_pred = knn_clf_pred))


# Cross Validating the knn model with three folds
cv_knn_3 = cross_val_score(knn_clf,
                           datagt,
                           datagt_target,
                           cv = 3)


#print(cv_knn_3)


#print(pd.np.mean(cv_knn_3).round(3))

#print('\nAverage: ',
 #     pd.np.mean(cv_knn_3).round(3),
  #    '\nMinimum: ',
   #   min(cv_knn_3).round(3),
    #  '\nMaximum: ',
     # max(cv_knn_3).round(3))


# Cross Validating the knn model with three folds
cv_knn_5 = cross_val_score(knn_clf,
                           datagt,
                           datagt_target,
                           cv = 5)


#print(cv_knn_5)


#print(pd.np.mean(cv_knn_5).round(3))

#print('\nAverage: ',
 #     pd.np.mean(cv_knn_5).round(3),
  #    '\nMinimum: ',
   #   min(cv_knn_5).round(3),
    #  '\nMaximum: ',
     # max(cv_knn_5).round(3))


# Cross Validating the knn model with 10 folds
cv_knn_10 = cross_val_score(knn_clf,
                           datagt,
                           datagt_target,
                           cv = 10)


#print(cv_knn_10)


#print(pd.np.mean(cv_knn_10).round(3))

#print('\nAverage: ',
 #     pd.np.mean(cv_knn_10).round(3),
  #    '\nMinimum: ',
   #   min(cv_knn_10).round(3),
    #  '\nMaximum: ',
     # max(cv_knn_10).round(3))


# Test on 17 neighbors, algorithm = 'brute' and weights = 'uniform', resulting from Hyperparameter tuning
knn_clf = KNeighborsClassifier(n_neighbors = 17, algorithm = 'brute', weights='uniform')


# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)


# Let's compare the testing score to the training score.
#print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
#print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


# Generating Predictions based on the optimal KNN model
knn_clf_pred = knn_clf_fit.predict(X_test)

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)


# Cross Validating the knn model with three folds
cv_knn_3 = cross_val_score(knn_clf,
                           datagt,
                           datagt_target,
                           cv = 3)


print(cv_knn_3)


print(pd.np.mean(cv_knn_3).round(3))

print('\nAverage: ',
      pd.np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
     '\nMaximum: ',
      max(cv_knn_3).round(3))


# Cross Validating the knn model with three folds
cv_knn_5 = cross_val_score(knn_clf,
                           datagt,
                           datagt_target,
                           cv = 5)


#print(cv_knn_5)


#print(pd.np.mean(cv_knn_5).round(3))

#print('\nAverage: ',
    #  pd.np.mean(cv_knn_5).round(3),
   #   '\nMinimum: ',
  #    min(cv_knn_5).round(3),
 #     '\nMaximum: ',
#      max(cv_knn_5).round(3))


# Cross Validating the knn model with three folds
cv_knn_10 = cross_val_score(knn_clf,
                           datagt,
                           datagt_target,
                           cv = 10)


#print(cv_knn_10)


#print(pd.np.mean(cv_knn_10).round(3))

#print('\nAverage: ',
 #     pd.np.mean(cv_knn_10).round(3),
  #    '\nMinimum: ',
   #   min(cv_knn_10).round(3),
    #  '\nMaximum: ',
     # max(cv_knn_10).round(3))

# Print classification report

#print(classification_report(y_true = y_test,
#                            y_pred = knn_clf_pred))



##### By testing both of Hyperparameter tuning result and best neighbor   #####
##### result, surprisingly, no significant difference between these two.  ##### 


#########################################
##### Building Classification Trees #####
#########################################

c_tree = DecisionTreeClassifier(random_state = 710)

c_tree_fit = c_tree.fit(X_train, y_train)


#print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
#print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))


##### Optimizing for hyperparameter #####

# Creating a hyperparameter grid

depth_space = pd.np.arange(1, 50)
criterion_space = ['gini', 'entropy']
leaf_space = pd.np.arange(1, 500)


param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space}


# Building the model object one more time

c_tree_1_hp = DecisionTreeClassifier(random_state = 710)


# Creating a GridSearchCV object

c_tree_1_hp_cv = GridSearchCV(c_tree_1_hp, param_grid, n_jobs = -1, cv = 3)


# Fit it to the training data

c_tree_1_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score

#print("Tuned Classification tree Parameter:", c_tree_1_hp_cv.best_params_)
#print("Tuned Classification tree Accuracy:", c_tree_1_hp_cv.best_score_.round(4))


# Building a tree model object with optimal hyperparameters

c_tree_optimal = DecisionTreeClassifier(criterion = 'entropy',
                                        random_state = 710,
                                        max_depth = 5,
                                        min_samples_leaf = 1)


c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)


#print('Training Score', c_tree_optimal_fit.score(X_train, y_train).round(4))
#print('Testing Score:', c_tree_optimal_fit.score(X_test, y_test).round(4))

# Plot importance of factors 
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
#plot_feature_importances(c_tree_optimal,
#                         train = X_train)


# Building a tree model object with optimal hyperparameters
c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 5,
                                        min_samples_leaf = 10)


c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)


dot_data = StringIO()


export_graphviz(decision_tree = c_tree_optimal_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train.columns)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#Image(graph.create_png(),
 #     height = 500,
  #    width = 800)


####################################
#####   Random_Forest model    #####
####################################

# Full forest using gini

forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


# Fitting the models

gt_fit = forest_gini.fit(X_train, y_train)


# Scoring the gini model

#print('Training Score', gt_fit.score(X_train, y_train).round(4))


#print('Testing Score:', gt_fit.score(X_test, y_test).round(4))


# Creating a hyperparameter grid

estimator_space = pd.np.arange(100, 2350, 250)


leaf_space = pd.np.arange(1, 150, 15)


criterion_space = ['gini', 'entropy']


bootstrap_space = [True, False]


warm_start_space = [True, False]


param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}


# Building the model object one more time

full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 710)


# Creating a GridSearchCV object

full_forest_cv = GridSearchCV(full_forest_grid, param_grid, n_jobs = -1, cv = 3)


# Fit it to the training data

full_forest_cv.fit(X_train, y_train)


# Print the optimal parameters and best score

#print("Tuned Random Forest Parameter:", full_forest_cv.best_params_)
#print("Tuned Random Forest Accuracy:", full_forest_cv.best_score_.round(4))


##### Building Random Forest Model Based on Best Parameters #####

rf_optimal = RandomForestClassifier(bootstrap = True,
                                    criterion = 'entropy',
                                    min_samples_leaf = 1,
                                    n_estimators = 350,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


#print('Training Score', rf_optimal.score(X_train, y_train).round(4))
#print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train  = rf_optimal.score(X_test, y_test)
rf_optimal_test = rf_optimal.score(X_train, y_train)


##### ROC & AUC #####

# Compute ROC curve and ROC area for each class

fpr, tpr, threshold = roc_curve(y_test, rf_optimal_pred)


roc = roc_auc_score(y_test, rf_optimal_pred)

#plt.title('ROC Validation')
#plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc)
#plt.legend(loc='lower right')
#plt.plot([0, 1], [0, 1], 'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()


# Cross Validating the RF model with three folds
cv_rf_3 = cross_val_score(rf_optimal,
                           datagt,
                           datagt_target,
                           cv = 3)


print(cv_rf_3)


print(pd.np.mean(cv_rf_3).round(3))

print('\nAverage: ',
      pd.np.mean(cv_rf_3).round(3),
      '\nMinimum: ',
      min(cv_rf_3).round(3),
      '\nMaximum: ',
      max(cv_rf_3).round(3))


#####################################
##### Gradient Boosted Machines #####
#####################################

gbm_3 = GradientBoostingRegressor(loss = 'ls',
                                   learning_rate = 0.1,
                                   n_estimators = 600,
                                   max_depth = 10,
                                   min_samples_leaf = 1,
                                   criterion = 'friedman_mse',
                                   warm_start = False,
                                   random_state = 710
                                   )



gbm_basic_fit = gbm_3.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_basic_fit.score(X_train, y_train)
gmb_basic_test  = gbm_basic_fit.score(X_test, y_test)


##### Applying GridSearchCV to Blueprint GBM #####

# Creating a hyperparameter grid
learn_space = pd.np.arange(0.01, 2.01, 0.05)
estimator_space = pd.np.arange(50, 1000, 50)
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'n_estimators' : estimator_space,
              'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space}



# Building the model object one more time
gbm_grid = GradientBoostingRegressor(random_state = 508)



# Creating a GridSearchCV object
gbm_grid_cv = RandomizedSearchCV(estimator = gbm_grid,
                                 param_distributions = param_grid,
                                 n_iter = 50,
                                 scoring = None,
                                 n_jobs = -1,
                                 cv = 3,
                                 random_state = 508)



# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))

### Even by using the most powerful computer I have, Alienware 17", which should be 
### powerful enough, it takes me 17 hours to run the grid search, thus still no 
### result had came out. So, no model is successfully created by me.

##########################
##### Saving Results #####
##########################

# Saving best model scores
model_scores_df = pd.DataFrame({'KNN_Score': cv_knn_3,
                                'RF_Score': rf_optimal_train})


model_scores_df.to_excel("Ensemble_Model_Results.xlsx")



# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_clf_pred,
                                     'RF_Predicted': rf_optimal_pred})


model_predictions_df.to_excel("Ensemble_Model_Predictions.xlsx")























