#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:20:23 2019

@author: bodmeryao
"""

# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



# Importing dataset
customers_df = pd.read_excel('finalExam_Mobile_App_Survey_Data.xlsx')

###############################################################################
# PCA
###############################################################################



########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2:77 ]

# Q50 cleanning up 

customers_df['children'] = customers_df['q50r2'] + customers_df['q50r3'] + customers_df['q50r4'] + customers_df['q50r5']

fill = 0


########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)


plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Wholesale Customer Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 7,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(customers_df.columns[2:77])


print(factor_loadings_df)


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')


########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)


########################
# Step 8: Rename your principal components and reattach demographic information
########################

X_pca_df.columns = ['Entertainment', 'Productive', 'Tooler', 'Modern', 'Moneyspender', 'Follower', 'Youngsters']
# Entertainment group are those who like shopping, trying new things and being a opinion leader. Those are mobile-head
# Productive group are those who depand on internet to get connected with families and to improve productivity.
# Tooler group are those who use mobile as a tool for everything
# Modern are mostly iphone user that living a modern life, not only online.
# Moneyspender group are those people who would willing to spend money for things they like and they have purchasing power
# Follower group are people who listen to others cool idea and don't really care about tech improvement. 
# Youngster group are assumed to be younger people into entertainment who don't really have money, though they are willing to pay for service.


final_pca_df = pd.concat([customers_df.loc[ : , ['q1','q48', 'q49','q50r1','q50r2','q50r3','q50r4','q50r5','children','q54','q55','q56','q57']] , X_pca_df], axis = 1)


########################
# Step 9: Analyze in more detail
########################


# Renaming age
age_names = {1 : 'Under 18',
                 2 : '18-24',
                 3 : '25-29',
                 4 : '30-34',
                 5 : '35-39',
                 6 : '40-44',
                 7 : '45-49',
                 8 : '50-54',
                 9 : '55-59',
                 10 : '60-64',
                 11 : '65 or over'}


final_pca_df['q1'].replace(age_names, inplace = True)


# Renaming education level
edu_names = {1 : 'Some high school',
             2 : 'High school graduate',
             3 : 'Some collage',
             4 : 'Collage graduate',
             5 : 'Some post-graduate studies',
             6 : 'Post graduate degree'}


final_pca_df['q48'].replace(edu_names, inplace = True)


# Renaming marital status
ms_names = {1 : 'Married',
             2 : 'Single',
             3 : 'Single with a partner',
             4 : 'Separated/Widowed/Divorced'}


final_pca_df['q49'].replace(ms_names, inplace = True)


# Renaming race
race_names = {1 : 'White or Caucasian',
              2 : 'Black or African American',
              3 : 'Asian',
              4 : 'Native Hawaiian or Other Pacific Islander',
              5 : 'American Indian or Alaska Native',
              6 : 'Other race'}


final_pca_df['q54'].replace(race_names, inplace = True)


# Renaming income
income_names = {1 : 'Under 10,000',
                2 : '10,000-14,999',
                3 : '15,000-19,999',
                4 : '20,000-29,999',
                5 : '30,000-39,999',
                6 : '40,000-49,999',
                7 : '50,000-59,999',
                8 : '60,000-69,999',
                9 : '70,000-79,999',
                10 : '80,000-89,999',
                11 : '90,000-99,999',
                12 : '100,000-124,999',
                13 : '125,000-149,999',
                14 : '150,000 or over'}


final_pca_df['q56'].replace(income_names, inplace = True)


# Renaming ethnicity
eth_names = {1 : 'Hispanic or Latino',
              2 : 'Not Hispanic or Latino'}


final_pca_df['q55'].replace(eth_names, inplace = True)


# Renaming gender
gender_names = {1 : 'Male',
                2 : 'Female'}


final_pca_df['q57'].replace(gender_names, inplace = True)


# Renaming group of children have
child_names = {0 : 'No kid',
               1 : 'One kid',
               2 : 'More than one kid',
               3 : 'More than one kid',
               4 : 'More than one kid'}


final_pca_df['children'].replace(child_names, inplace = True)


# Analyzing the Productive group
# Race
fig, ax = plt.subplots(figsize = (12, 4))
sns.boxplot(x = 'q54',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# Gender
fig, ax = plt.subplots(figsize = (4, 4))
sns.boxplot(x = 'q57',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# Age
fig, ax = plt.subplots(figsize = (16, 4))
sns.boxplot(x = 'q1',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# Marital status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# Education level
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


# Childrens situation
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'children',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize = (4, 4))
sns.boxplot(x = 'q50r5',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()

# Race
fig, ax = plt.subplots(figsize = (12, 4))
sns.boxplot(x = 'q54',
            y =  'Productive',
            data = final_pca_df)

plt.ylim(-2, 4)
plt.tight_layout()
plt.show()



###############################################################################
# Cluster Analysis 
###############################################################################

from sklearn.cluster import KMeans # k-means clustering


########################
# Step 1: Remove demographic information
########################

customer_features_reduced = customers_df.iloc[ : , 2:77 ]



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k = KMeans(n_clusters = 10,
                      random_state = 508)


customers_k.fit(X_scaled_reduced)


customers_kmeans_clusters = pd.DataFrame({'cluster': customers_k.labels_})


print(customers_kmeans_clusters.iloc[: , 0].value_counts())



########################
# Step 4: Analyze cluster centers
########################

centroids = customers_k.cluster_centers_


centroids_df = pd.DataFrame(centroids)



# Renaming columns
centroids_df.columns = customer_features_reduced.columns


print(centroids_df)


# Sending data to Excel
centroids_df.to_excel('customers_k3_centriods.xlsx')


########################
# Step 5: Analyze cluster memberships
########################


X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


X_scaled_reduced_df.columns = customer_features_reduced.columns


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)


print(clusters_df)



########################
# Step 6: Reattach demographic information 
########################

final_clusters_df = pd.concat([customers_df.loc[ : , ['q1','q48', 'q49','q50r1','q50r2','q50r3','q50r4','q50r5','children','q54','q55','q56','q57'] ],
                               clusters_df],
                               axis = 1)


print(final_clusters_df)



########################
# Step 7: Analyze in more detail 
########################


# Renaming age
age_names = {1 : 'Under 18',
                 2 : '18-24',
                 3 : '25-29',
                 4 : '30-34',
                 5 : '35-39',
                 6 : '40-44',
                 7 : '45-49',
                 8 : '50-54',
                 9 : '55-59',
                 10 : '60-64',
                 11 : '65 or over'}


final_clusters_df['q1'].replace(age_names, inplace = True)


# Renaming education level
edu_names = {1 : 'Some high school',
             2 : 'High school graduate',
             3 : 'Some collage',
             4 : 'Collage graduate',
             5 : 'Some post-graduate studies',
             6 : 'Post graduate degree'}


final_clusters_df['q48'].replace(edu_names, inplace = True)


# Renaming marital status
ms_names = {1 : 'Married',
             2 : 'Single',
             3 : 'Single with a partner',
             4 : 'Separated/Widowed/Divorced'}


final_clusters_df['q49'].replace(ms_names, inplace = True)


# Renaming race
race_names = {1 : 'White or Caucasian',
              2 : 'Black or African American',
              3 : 'Asian',
              4 : 'Native Hawaiian or Other Pacific Islander',
              5 : 'American Indian or Alaska Native',
              6 : 'Other race'}


final_clusters_df['q54'].replace(race_names, inplace = True)


# Renaming income
income_names = {1 : 'Under 10,000',
                2 : '10,000-14,999',
                3 : '15,000-19,999',
                4 : '20,000-29,999',
                5 : '30,000-39,999',
                6 : '40,000-49,999',
                7 : '50,000-59,999',
                8 : '60,000-69,999',
                9 : '70,000-79,999',
                10 : '80,000-89,999',
                11 : '90,000-99,999',
                12 : '100,000-124,999',
                13 : '125,000-149,999',
                14 : '150,000 or over'}


final_clusters_df['q56'].replace(income_names, inplace = True)


# Renaming ethnicity
eth_names = {1 : 'Hispanic or Latino',
              2 : 'Not Hispanic or Latino'}


final_clusters_df['q55'].replace(eth_names, inplace = True)


# Renaming gender
gender_names = {1 : 'Male',
                2 : 'Female'}


final_clusters_df['q57'].replace(gender_names, inplace = True)


# Renaming group of children have
child_names = {0 : 'No kid',
               1 : 'One kid',
               2 : 'More than one kid',
               3 : 'More than one kid',
               4 : 'More than one kid'}


final_clusters_df['children'].replace(child_names, inplace = True)


###############################################################################
# Combining PCA and Clustering!!!
###############################################################################


########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())




########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Entertainment', 'Productive', 'Tooler', 'Modern', 'Moneyspender', 'Follower', 'Youngsters']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([customers_df.loc[ : , ['q1','q48', 'q49','q50r1','q50r2','q50r3','q50r4','q50r5','children','q54','q55','q56','q57'] ],
                                clst_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))


########################
# Step 7: Analyze in more detail 
########################


# Renaming age
age_names = {1 : 'Under 18',
                 2 : '18-24',
                 3 : '25-29',
                 4 : '30-34',
                 5 : '35-39',
                 6 : '40-44',
                 7 : '45-49',
                 8 : '50-54',
                 9 : '55-59',
                 10 : '60-64',
                 11 : '65 or over'}


final_pca_clust_df['q1'].replace(age_names, inplace = True)


# Renaming education level
edu_names = {1 : 'Some high school',
             2 : 'High school graduate',
             3 : 'Some collage',
             4 : 'Collage graduate',
             5 : 'Some post-graduate studies',
             6 : 'Post graduate degree'}


final_pca_clust_df['q48'].replace(edu_names, inplace = True)


# Renaming marital status
ms_names = {1 : 'Married',
             2 : 'Single',
             3 : 'Single with a partner',
             4 : 'Separated/Widowed/Divorced'}


final_pca_clust_df['q49'].replace(ms_names, inplace = True)


# Renaming race
race_names = {1 : 'White or Caucasian',
              2 : 'Black or African American',
              3 : 'Asian',
              4 : 'Native Hawaiian or Other Pacific Islander',
              5 : 'American Indian or Alaska Native',
              6 : 'Other race'}


final_pca_clust_df['q54'].replace(race_names, inplace = True)


# Renaming income
income_names = {1 : 'Under 10,000',
                2 : '10,000-14,999',
                3 : '15,000-19,999',
                4 : '20,000-29,999',
                5 : '30,000-39,999',
                6 : '40,000-49,999',
                7 : '50,000-59,999',
                8 : '60,000-69,999',
                9 : '70,000-79,999',
                10 : '80,000-89,999',
                11 : '90,000-99,999',
                12 : '100,000-124,999',
                13 : '125,000-149,999',
                14 : '150,000 or over'}


final_pca_clust_df['q56'].replace(income_names, inplace = True)


# Renaming ethnicity
eth_names = {1 : 'Hispanic or Latino',
              2 : 'Not Hispanic or Latino'}


final_pca_clust_df['q55'].replace(eth_names, inplace = True)


# Renaming gender
gender_names = {1 : 'Male',
                2 : 'Female'}


final_pca_clust_df['q57'].replace(gender_names, inplace = True)


# Renaming group of children have
child_names = {0 : 'No kid',
               1 : 'One kid',
               2 : 'More than one kid',
               3 : 'More than one kid',
               4 : 'More than one kid'}


final_pca_clust_df['children'].replace(child_names, inplace = True)


# Adding a productivity step
data_df = final_pca_clust_df

########################
# Productive
########################

# Race
fig, ax = plt.subplots(figsize = (16, 4))
sns.boxplot(x = 'q54',
            y = 'Productive',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 8)
plt.tight_layout()
plt.show()



# Marital status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y = 'Productive',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()



# Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y = 'Productive',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()

