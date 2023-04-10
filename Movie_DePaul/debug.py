from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from stellargraph.mapper import Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec
# from tensorflow.keras import Model, optimizers, losses
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


df = pd.read_csv('Movie_DePaulMovie/ratings.txt',sep=',')
enconder = LabelEncoder()
df['itemid'] = enconder.fit_transform(df['itemid'])
print(df['Location'].unique())
print(df['Companion'].unique())
print(df['Time'].unique())
map_location= {'Cinema':1, 'Home':2}
map_time= {'Weekday':1, 'Weekend':2}
map_com= {'Alone':1, 'Family':2, 'Partner':3}
df['Location'] = df['Location'].map(map_location)
df['Companion'] = df['Companion'].map(map_com)
df['Time'] = df['Time'].map(map_time)
location = df[['userid','Location']]
group_location = pd.get_dummies(location).groupby('userid').max().reset_index()
for i in group_location.userid: 
    a = group_location[group_location['userid']==i]
    if a['Location_Cinema'].values[0] == 0 and a['Location_Home'].values[0] == 0: 
        group_location[group_location['userid']==i]['Location_Cinema']== np.nan
        group_location[group_location['userid']==i]['Location_Home']== np.nan
imputer = KNNImputer(n_neighbors=5)
imputed_df = pd.DataFrame(imputer.fit_transform(df),columns=['userid', 'itemid', 'rating', 'Time', 'Location', 'Companion'])
for i in imputed_df.columns: 
    imputed_df[i] = imputed_df[i].astype('int')

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming your dataframe is called df
X = imputed_df[['Time', 'Location', 'Companion','userid']]
y = imputed_df['rating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the LinearSVC model
svm = LinearSVC(random_state=42)
svm.fit(X_train_scaled, y_train)

# Get feature importances
importances = svm.coef_

# Print the feature importances
features = ['Time', 'Location', 'Companion','userid']
for i in range(importances.shape[0]):
    print(f'Class {i+1}:')
    for j in range(importances.shape[1]):
        print(f'  {features[j]}: {importances[i, j]}')

# Calculate the mean importance for each feature across all classes
mean_importances = np.mean(importances, axis=0)

# Create a dictionary to map feature names to mean importances
feature_importance_dict = {feature: importance for feature, importance in zip(features, mean_importances)}

# Replace the values in 'Time', 'Location', and 'Companion' columns with their respective feature importances
df_replaced = imputed_df.copy()
for feature in features:
    df_replaced[feature] = df_replaced[feature] * feature_importance_dict[feature]

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier

# X = imputed_df[['Time', 'Location', 'Companion']]
# y = imputed_df['rating']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Fit the AdaBoost model with a DecisionTreeClassifier as the base estimator
# base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
# ada = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)
# ada.fit(X_train_scaled, y_train)

# # Get feature importances
# importances = ada.feature_importances_

# # Print the feature importances
# features = ['Time', 'Location', 'Companion']
# for i, importance in enumerate(importances):
#     print(f'{features[i]}: {importance}')
X_ = imputed_df[['Time', 'Location', 'Companion','itemid']]
y_ = imputed_df['rating']
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the LinearSVC model
svm = LinearSVC(random_state=42)
svm.fit(X_train_scaled, y_train)

# Get feature importances
importances_ = svm.coef_

# Print the feature importances
features_ = ['Time', 'Location', 'Companion','itemid']
for i in range(importances_.shape[0]):
    print(f'Class {i+1}:')
    for j in range(importances_.shape[1]):
        print(f'  {features_[j]}: {importances_[i, j]}')

# Calculate the mean importance for each feature across all classes
mean_importances = np.mean(importances_, axis=0)

# Create a dictionary to map feature names to mean importances
feature_importance_dict = {feature: importance for feature, importance in zip(features_, mean_importances)}

# Replace the values in 'Time', 'Location', and 'Companion' columns with their respective feature importances
df_replaced_ = imputed_df.copy()
for feature in features_:
    df_replaced_[feature] = df_replaced_[feature] * feature_importance_dict[feature]
imputed_df['Time_user'] = df_replaced['Time']
imputed_df['Location_user']= df_replaced['Location']
imputed_df['Companion_user']= df_replaced['Companion']
imputed_df['Time_item'] = df_replaced_['Time']
imputed_df['Location_item']= df_replaced_['Location']
imputed_df['Companion_item']= df_replaced_['Companion']


global_mean = df_replaced["rating"].mean()
user_bias = df_replaced.groupby("userid")["rating"].mean() - global_mean
item_bias = df_replaced.groupby("itemid")["rating"].mean() - global_mean

from sklearn.decomposition import NMF
model = NMF(n_components=5, init='random', random_state=0) #n_component = KNN values
U_matrix = model.fit_transform(rating_matrix.fillna(0).values)
V_matrix = model.components_