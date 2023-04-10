#!/usr/bin/env python
# coding: utf-8

# In[20]:

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from itertools import product, permutations
from multiprocessing import Pool
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import warnings
import random
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
np.random.seed(42)
scaler_minmax=MinMaxScaler()


# In[21]:


df = pd.read_csv('C:/Users/thanh/OneDrive - Queensland University of Technology/dataPHD/Music_InCarMusic/Music_InCarMusic/music_full.csv').drop(columns=['Unnamed: 0'])
origin_data= pd.read_excel('C:/Users/thanh/OneDrive - Queensland University of Technology/dataPHD/Music_InCarMusic/Music_InCarMusic/Data_InCarMusic.xlsx',sheet_name=0)
df_1= pd.read_excel('C:/Users/thanh/OneDrive - Queensland University of Technology/dataPHD/Music_InCarMusic/Music_InCarMusic/Data_InCarMusic.xlsx',sheet_name=1)
df_2= pd.read_excel('C:/Users/thanh/OneDrive - Queensland University of Technology/dataPHD/Music_InCarMusic/Music_InCarMusic/Data_InCarMusic.xlsx',sheet_name=2)
df_3= pd.read_excel('C:/Users/thanh/OneDrive - Queensland University of Technology/dataPHD/Music_InCarMusic/Music_InCarMusic/Data_InCarMusic.xlsx',sheet_name=3)
origin_data=origin_data.fillna(0)


# In[22]:


# from itertools import product

# def grid_search(model, param_grid, n_iter=300):
#     best_loss = float('inf')
#     best_params = None
#     for params in product(*param_grid.values()):
#         params = dict(zip(param_grid.keys(), params))
#         mf = model(**params)
#         mf.factorize(iter=n_iter)
#         loss = mf.total_loss[-1]
#         if loss < best_loss:
#             best_loss = loss
#             best_params = params
#     print('Best parameters:', best_params)
#     print('Best loss:', best_loss)
#     return best_params


def grid_search(model, param_grid, test, filename='music_grid_search_results.xlsx'):
    best_loss = float('inf')
    best_params = None
    results = []
    for params in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), params))
        mf = model(**params)
        mf.evaluation(test)
        loss = mf.total_loss[-1]
        results.append({**params, 'loss': loss})
        if loss < best_loss:
            best_loss = loss
            best_params = params
    print('Best parameters:', best_params)
    print('Best loss:', best_loss)
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    return best_params
# import pandas as pd
# from itertools import product
# from multiprocessing import Process, Queue, cpu_count

# def factorize_worker(model, params_list, n_iter, result_queue):
#     for params in params_list:
#         mf = model(**params)
#         mf.factorize(iter=n_iter)
#         loss = mf.total_loss[-1]
#         result_queue.put({**params, 'loss': loss})

# def grid_search(model, param_grid, n_iter=2, n_jobs=2, filename='grid_search_results.xlsx'):
#     if n_jobs == -1:
#         n_jobs = cpu_count()

#     best_loss = float('inf')
#     best_params = None
#     results = []

#     params_list = [dict(zip(param_grid.keys(), p)) for p in product(*param_grid.values())]
#     result_queue = Queue()

#     processes = []
#     chunk_size = (len(params_list) + n_jobs - 1) // n_jobs

#     for i in range(n_jobs):
#         start = i * chunk_size
#         end = min((i + 1) * chunk_size, len(params_list))
#         p = Process(target=factorize_worker, args=(model, params_list[start:end], n_iter, result_queue))
#         processes.append(p)

#     for p in processes:
#         p.start()

#     for _ in range(len(params_list)):
#         results.append(result_queue.get())

#     for p in processes:
#         p.join()

#     df = pd.DataFrame(results)
#     df.to_excel(filename, index=False)

#     best_params = df.loc[df['loss'].idxmin()].to_dict()
#     best_loss = best_params.pop('loss')

#     print('Best parameters:', best_params)
#     print('Best loss:', best_loss)

#     return best_params


# In[23]:


def cal(df):

    
    # calculate total number of possible user-item interactions
    num_users = df[df.columns[0]].nunique()
    num_items = df[df.columns[1]].nunique()
    num_possible_interactions = num_users * num_items
    
    # calculate total number of actual user-item interactions
    num_actual_interactions = df.shape[0]
    
    # calculate sparsity of ratings
    sparsity = 1 - (num_actual_interactions / num_possible_interactions)
    
    print(sparsity)
cal(df)


# In[24]:


def count_nan(df):
    """
    Returns the percentage of NaN values in a pandas DataFrame.
    """
    total_cells = df.size
    nan_cells = df.isna().sum().sum()
    nan_percentage = (nan_cells / total_cells) * 100
    print(nan_percentage)
count_nan(df)


# In[25]:


# neighbors=int(df['userid'].value_counts().mean())
# imputer = KNNImputer(n_neighbors=neighbors)
# imputed_df = pd.DataFrame(imputer.fit_transform(df),columns=['userid', 'itemid', 'rating', 'Time', 'Location', 'Companion'])

# for i in imputed_df.columns: 
#     imputed_df[i] = imputed_df[i].astype('int')
# imputed_df
df.columns


# # Generating Contextual Coeficient

# In[26]:




# Assuming your dataframe is called df
X = df[['UserID', 'DrivingStyle', 'landscape', 'mood',
       'naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions',
       'weather', ]]
y = df['Rating']

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
features = ['UserID', 'DrivingStyle', 'landscape', 'mood',
       'naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions',
       'weather', ]
for i in range(importances.shape[0]):
    print(f'Class {i+1}:')
    for j in range(importances.shape[1]):
        print(f'  {features[j]}: {importances[i, j]}')

# Calculate the mean importance for each feature across all classes
mean_importances = np.mean(importances, axis=0)

# Create a dictionary to map feature names to mean importances
feature_importance_dict = {feature: importance for feature, importance in zip(features, mean_importances)}

# Replace the values in 'Time', 'Location', and 'Companion' columns with their respective feature importances
df_replaced =df.copy()
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

X = df[['ItemID', 'DrivingStyle', 'landscape', 'mood',
       'naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions',
       'weather', ]]
y = df['Rating']
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
features = ['ItemID', 'DrivingStyle', 'landscape', 'mood',
       'naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions',
       'weather', ]
for i in range(importances.shape[0]):
    print(f'Class {i+1}:')
    for j in range(importances.shape[1]):
        print(f'  {features[j]}: {importances[i, j]}')

# Calculate the mean importance for each feature across all classes
mean_importances = np.mean(importances, axis=0)

# Create a dictionary to map feature names to mean importances
feature_importance_dict = {feature: importance for feature, importance in zip(features, mean_importances)}

# Replace the values in 'Time', 'Location', and 'Companion' columns with their respective feature importances
df_replaced_item =df.copy()
for feature in features:
    df_replaced_item[feature] = df_replaced_item[feature] * feature_importance_dict[feature]


# In[27]:


df_fix_user = df_replaced[[ 'DrivingStyle', 'landscape', 'mood',
       'naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions',
       'weather', ]]
df


# In[28]:


df_fix_user = df_replaced[['DrivingStyle', 'landscape', 'mood',
       'naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions',
       'weather', ]]
for i in ['DrivingStyle', 'landscape', 'mood','naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions','weather', ]:
    for j in range(len(df_fix_user[i])):
        if origin_data[i][j] ==0: 
            df_replaced[i][j]=0
df_fix_user = df_replaced_item[['DrivingStyle', 'landscape', 'mood','naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions','weather', ]]
for i in ['DrivingStyle', 'landscape', 'mood','naturalphenomena ', 'RoadType', 'sleepiness', 'trafficConditions','weather', ]:
    for j in range(len(df_fix_user[i])):
        if origin_data[i][j] ==0: 
            df_replaced_item[i][j]=0      
for i in df_fix_user.columns: 
    prefix = "_user"
    # create a new column name by adding the prefix to the original column name
    new_col = i+prefix
    # create the new column by copying the original column data to the new column
    df[new_col] = df_replaced[i]
for i in df_fix_user.columns: 
    prefix = "_item"
    # create a new column name by adding the prefix to the original column name
    new_col = i+prefix
    # create the new column by copying the original column data to the new column
    df[new_col] = df_replaced_item[i]
try:
    df=df.drop(columns=['Unnamed: 0'])
except KeyError:
    pass


# In[29]:


df


# # Get Bias

# In[30]:


df_replaced.head(30)
global_mean = df_replaced["Rating"].mean()
user_bias = df_replaced.groupby("UserID")["Rating"].mean() - global_mean
item_bias = df_replaced.groupby("ItemID")["Rating"].mean() - global_mean
print(np.min(user_bias))
print(np.max(user_bias))
print(np.mean(user_bias))
print("SUM: ",np.sum(user_bias))
print(np.min(item_bias))
print(np.max(item_bias))
print(np.mean(item_bias))
print("SUM: ",np.sum(item_bias))


# In[31]:



rating_matrix_original= df[['UserID','ItemID','Rating']].pivot_table(values='Rating',index='UserID',columns='ItemID', fill_value=0).astype('int')
rating_matrix_original.values
df_2.columns


# # Generate real Laplacian

# In[32]:



def to_int(x):
    if pd.isna(x):
        return x
    return int(x)
rating_matrix=rating_matrix_original
df_onehot = pd.get_dummies(df_2[['category_id', ' artist']])
similarity_matrix = cosine_similarity(df_onehot)
def calculate_L(similarity_matrix):
    L = []

    for i in range(len(similarity_matrix)):
        a = 0
        for j in range(len(similarity_matrix)):
            a = similarity_matrix[i][j] * np.sum((similarity_matrix[i] - similarity_matrix[j]))
        L.append(a)
    return L
df.columns


# # Get Laplacian

# In[33]:


neighbors=52

# Compute adjacency matrices based on similarity
# user_adj_matrix = adjacency_matrix_similarity(U_matrix)
# item_adj_matrix = adjacency_matrix_similarity(V_matrix)
# # Compute Laplacian matrices
# L_U = laplacian_matrix(user_adj_matrix)
# L_V = laplacian_matrix(item_adj_matrix)
# Get context matrix
scaler = StandardScaler()
# minmax = MinMaxScaler(feature_range=(0,2))
Cu=df[['DrivingStyle_user', 'landscape_user', 'mood_user','naturalphenomena _user', 'RoadType_user', 'sleepiness_user','trafficConditions_user', 'weather_user']].multiply(100000)
Ci=df[['DrivingStyle_item',	'landscape_item', 'mood_item', 'naturalphenomena _item','RoadType_item', 'sleepiness_item', 'trafficConditions_item','weather_item']].multiply(100000)
C_umatrix =  np.array(Cu)
C_imatrix =  np.array(Ci)
print(np.min(C_imatrix))
print(np.max(C_imatrix))
print(np.median(C_imatrix))
print("SUM: ",np.sum(C_imatrix))
print(np.min(C_umatrix))
print(np.max(C_umatrix))
print(np.median(C_umatrix))
print("SUM: ",np.sum(C_umatrix))


# In[34]:


def rmse(predicted_ratings, real_ratings):
    # Create a mask of the same shape as real_ratings with True where there's a rating and False where there's NaN
    mask = ~np.isnan(real_ratings)

    # Calculate the squared error between the predicted and real ratings only for the rated items
    squared_error = (predicted_ratings[mask] - real_ratings[mask])**2

    # Calculate the mean squared error
    mean_squared_error = np.mean(squared_error)

    # Calculate the root mean squared error
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error

def mae(predicted_ratings, real_ratings):
    # Create a mask of the same shape as real_ratings with True where there's a rating and False where there's NaN
    mask = ~np.isnan(real_ratings)

    # Calculate the absolute error between the predicted and real ratings only for the rated items
    absolute_error = np.abs(predicted_ratings[mask] - real_ratings[mask])

    # Calculate the mean absolute error
    mean_absolute_error = np.mean(absolute_error)

    return mean_absolute_error
def top_10_f1_score(predicted_ratings, real_ratings):
    f1_sum = 0
    users_count = real_ratings.shape[0]

    for user_idx in range(users_count):
        user_real_ratings = real_ratings[user_idx]
        user_predicted_ratings = predicted_ratings[user_idx]

        # Get the indices of the top-10 predicted ratings
        top_10_predicted_indices = np.argsort(user_predicted_ratings)[-10:]

        # Get the indices of the user's real ratings
        real_rated_indices = np.where(~np.isnan(user_real_ratings))[0]

        # Calculate the number of relevant items in the top-10 predicted items
        relevant_items_count = np.sum(np.isin(top_10_predicted_indices, real_rated_indices))

        # Calculate the precision for the current user
        user_precision = relevant_items_count / 10

        # Calculate the recall for the current user
        user_recall = relevant_items_count / len(real_rated_indices)

        # Calculate the F1-score for the current user
        if user_precision + user_recall > 0:
            user_f1_score = 2 * user_precision * user_recall / (user_precision + user_recall)
        else:
            user_f1_score = 0

        # Update the F1-score sum
        f1_sum += user_f1_score

    # Calculate the average F1-score across all users
    average_f1_score = f1_sum / users_count

    return average_f1_score
def top_10_precision(predicted_ratings, real_ratings):
    precision_sum = 0
    users_count = real_ratings.shape[0]

    for user_idx in range(users_count):
        user_real_ratings = real_ratings[user_idx]
        user_predicted_ratings = predicted_ratings[user_idx]

        # Get the indices of the top-10 predicted ratings
        top_10_predicted_indices = np.argsort(user_predicted_ratings)[-10:]

        # Get the indices of the user's real ratings
        real_rated_indices = np.where(~np.isnan(user_real_ratings))[0]

        # Calculate the number of relevant items in the top-10 predicted items
        relevant_items_count = np.sum(np.isin(top_10_predicted_indices, real_rated_indices))

        # Calculate the precision for the current user
        user_precision = relevant_items_count / 10

        # Update the precision sum
        precision_sum += user_precision

    # Calculate the average precision across all users
    average_precision = precision_sum / users_count

    return average_precision

def top_1_precision(predicted_ratings, real_ratings):
    precision_sum = 0
    users_count = real_ratings.shape[0]

    for user_idx in range(users_count):
        user_real_ratings = real_ratings[user_idx]
        user_predicted_ratings = predicted_ratings[user_idx]

        # Get the indices of the top-10 predicted ratings
        top_10_predicted_indices = np.argsort(user_predicted_ratings)[-1:]

        # Get the indices of the user's real ratings
        real_rated_indices = np.where(~np.isnan(user_real_ratings))[0]

        # Calculate the number of relevant items in the top-10 predicted items
        relevant_items_count = np.sum(np.isin(top_10_predicted_indices, real_rated_indices))

        # Calculate the precision for the current user
        user_precision = relevant_items_count 

        # Update the precision sum
        precision_sum += user_precision

    # Calculate the average precision across all users
    average_precision = precision_sum / users_count

    return average_precision
def precision_at_k(predicted_ratings, real_ratings, k=10):
    precision_sum = 0
    users_count = real_ratings.shape[0]

    for user_idx in range(users_count):
        user_real_ratings = real_ratings[user_idx]
        user_predicted_ratings = predicted_ratings[user_idx]

        # Get the indices of the top-k predicted ratings
        top_k_predicted_indices = np.argsort(user_predicted_ratings)[-k:]

        # Get the indices of the user's real ratings
        real_rated_indices = np.where(~np.isnan(user_real_ratings))[0]

        # Calculate the number of relevant items in the top-k predicted items
        relevant_items_count = np.sum(np.isin(top_k_predicted_indices, real_rated_indices))

        # Calculate the precision for the current user
        user_precision = relevant_items_count / k

        # Update the precision sum
        precision_sum += user_precision

    # Calculate the average precision across all users
    average_precision = precision_sum / users_count

    return average_precision


# In[35]:




def k_fold_cross_validation(ratings, k=5, random_state=None):
    # Create a KFold object with the specified number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Initialize the list to store the train and test matrices for each fold
    train_test_matrices = []

    # Iterate over the splits
    for train_indices, test_indices in kf.split(ratings):
        # Copy the original ratings matrix for both train and test matrices
        train_matrix = ratings.copy()
        test_matrix = np.empty_like(ratings)

        test_matrix[:] = np.nan

        # Replace the test ratings with NaN in the train matrix and vice versa
        for row, col in zip(*np.nonzero(ratings)):
            if row in test_indices:
                train_matrix[row, col] = np.nan
                test_matrix[row, col] = ratings[row, col]

        # Add the train and test matrices to the list
        train_test_matrices.append((train_matrix, test_matrix))

    return train_test_matrices

class MultiMF: 
    def __init__(self,R_, P_, Q_,U_,V_,  E_, D_, C_u_,C_i_,  L_U_, L_V_, lambdas_, b_u_, b_v_,alpha,):

        self.R_matrix = torch.tensor(R_, dtype=torch.float32)
        self.P_matrix = torch.tensor(P_, dtype=torch.float32)
        self.Q_matrix = torch.tensor(Q_, dtype=torch.float32)
        self.lambdas = lambdas_
        self.U_matrix = torch.tensor(U_, dtype=torch.float32, requires_grad=True)
        self.V_matrix = torch.tensor(V_, dtype=torch.float32, requires_grad=True)
        self.E_matrix = torch.tensor(E_, dtype=torch.float32, requires_grad=True)
        self.D_matrix = torch.tensor(D_, dtype=torch.float32, requires_grad=True)
        self.C_umatrix = torch.tensor(C_u_, dtype=torch.float32)
        self.C_imatrix = torch.tensor(C_i_, dtype=torch.float32)
        self.L_U = torch.tensor(L_U_, dtype=torch.float32)
        self.L_V = torch.tensor(L_V_, dtype=torch.float32)
        self.item_bias = torch.tensor(b_v_, dtype=torch.float32)
        self.user_bias = torch.tensor(b_u_, dtype=torch.float32)
        self.alpha = alpha
        self.total_loss = [0]
    def factorize(self,iter=50):
        self.run_func()
        for k in tqdm(range(iter)):
            if abs(self.total_loss[-1] - self.total_loss[-2]) < 0.0004:
                print("Success")
                break
            else: 
                # self.U_matrix=np.array(self.newU)
                # self.V_matrix=np.array(self.newV).T
                # self.D_matrix=np.array(self.newD)
                # self.E_matrix=np.array(self.newE)
                # self.newU=[]
                # self.newV=[]
                # self.newE=[]
                # self.newD=[]
                self.run_func()
        # return self.U_matrix, self.V_matrix
    def run_func(self):
        optimizer = optim.SGD([self.U_matrix, self.V_matrix, self.E_matrix, self.D_matrix], lr=self.alpha)
        loss_col = 0
        lambdas = self.lambdas
        num_rows = len(self.R_matrix)
        sum_E = torch.sum(self.E_matrix)
        sum_P = torch.sum(self.P_matrix)
        Q_matrix_sum = torch.sum(self.Q_matrix)
        L_V_sum = torch.sum(self.L_V)
        sum_D = torch.sum(self.D_matrix)
        sum_LU = torch.sum(self.L_U)
        for i in range(num_rows):
            loss_row = 0
            U = self.U_matrix[i]
            C_u = self.C_umatrix[i, :]
            C_i = self.C_imatrix[i, :]
            b_u = self.user_bias[i]
            if sum_E != 0:
                E = self.E_matrix
            else:
                E = np.array(0)
            if sum_P != 0:
                P = self.P_matrix[i]
            else:
                P = np.array(0)

            if sum_D != 0:
                D = self.D_matrix
            else:
                D = np.array(0)

            if sum_LU != 0:
                Lu = self.L_U[i]
            else:
                Lu = np.array(0)
            V_T = self.V_matrix.T

            for j in range(len(self.R_matrix[i])):
                if not torch.isnan(self.R_matrix[i, j]):
                    optimizer.zero_grad()
                    R = self.R_matrix[i, j]
                    V = V_T[j]
                    if Q_matrix_sum != 0:
                        Q = self.Q_matrix[j]
                    else:
                        Q = np.array(0)
                    if L_V_sum != 0:
                        Li = self.L_V[j]
                    else:
                        Li = 0
                    # U, V, D, E = self.update_U_V_E_D(R=R, P=P, E=E, D=D, Q=Q, U=U, V=V, C_u=C_u, C_i=C_i, Lu=Lu, Li=Li, lambdas=lambdas)
      
                    b_v = self.item_bias[j]
                    loss= self.objective_function(R, P, Q, U, V, E, D, C_u, C_i, Lu, Li, lambdas, b_u, b_v)
 
                    # self.U_matrix[i] = torch.tensor(U)
                    # self.V_matrix[j] = V
                    # self.E_matrix[j] = E
                    loss.backward()
                    optimizer.step()

                    loss_row += loss.item()

            loss_col += loss_row
        self.total_loss.append(loss_col)

    def objective_function(self,R, P, Q, U, V, E, D, C_u,C_i,  L_U, L_V, lambdas, b_u, b_v):
        # U=U+0.0000000000005
        # V=V+0.0000000000005
        M = U.shape[0] #number of user
        V_=0
        U_=0
        
        U_ += torch.sum(C_u)*U
        V_ += torch.sum(C_i)*V
        U_mean=U_/M
        V_mean=V_/M
        R_pred = torch.dot(U_mean + U, V.T + V_mean)
        if torch.sum(self.P_matrix) != 0:
            P_pred = torch.dot(U, E.T)
        else:

            P_pred = 0
        if torch.sum(self.Q_matrix)!=0:
            Q_pred = torch.matmul(V.T, D)
        else:
            Q_pred=0
            
        loss = (R - R_pred)**2 + (P - P_pred)**2 + torch.sum((Q - Q_pred)**2)
        loss += lambdas[0] * torch.sum(U**2) + lambdas[1] * torch.sum(V**2) + lambdas[2] * torch.sum(D**2) + lambdas[3] * E**2
        loss += b_u + b_v
        
        return loss

    def update_U_V_E_D(self,R, P, Q, U, V, E, D, C_u,C_i,Lu,Li,lambdas):
        alpha=self.alpha
        # U=U+0.0000000000005
        # V=V+0.0000000000005
        M = U.shape[0] #number of user
        V_=0
        U_=0
        U_ += np.sum(C_u)*U
        V_ += np.sum(C_i)*V
        U_mean=U_/M

        V_mean=V_/M
        if np.sum(self.P_matrix) != 0:
            P_pred = np.dot(U, E.T)
        else:

            P_pred = 0
        if np.sum(self.Q_matrix)!=0:
            Q_pred = np.dot(V.T, D)
            print(Q_pred.shape)
        else:
            Q_pred=0
        R_pred = np.dot((U_mean + U), (V.T + V_mean))


        dU = -2 * (R- R_pred) * (V.T + V_mean)*(1+U_mean) - 2 * (P - P_pred) * E + 2 * lambdas[0] * U   + 2*lambdas[3]*Lu #change to lambda 5 soon
        print("R SHAPE: ",R_pred)
        print("U SHAPE: ",U.shape)
        print("V SHAPE: ",V.shape)
        print("D SHAPE: ",D.shape)
        print("Q SHAPE: ",Q.shape)
        print("Minus SHAPE: ",(Q - Q_pred).shape)
        print("Minus SHAPE 2: ",((Q.T - Q_pred.T)).shape)
        dV = -2 * (R - R_pred) * (U_mean + U)*(1+V_mean) - 2 * D*(Q - Q_pred) *  + 2 * lambdas[1] * V  + 2*lambdas[4]*Li

        dE = -2 * (P - P_pred) * U + 2 * lambdas[3] * E
        dD = -2 * (Q - Q_pred) * V + 2 * lambdas[4] * D
        
        U -= alpha * dU
        V -= alpha * dV
        E -= alpha * dE
        D -= alpha * dD


        return U,V,E,D

    def predict(self):
        self.factorize()
        mmscaler=MinMaxScaler(feature_range=(1, 6))
        R_pred = np.dot(self.U_matrix,self.V_matrix)
        R_pred=mmscaler.fit_transform(R_pred).astype(int)
        return R_pred
    def evaluation(self,test):
        import datetime

        R_pred = self.predict()
        print(R_pred)
        mae_value = mae(R_pred,test)
        rmse_value = rmse(R_pred,test)
        top_10_f1_score_value = top_10_f1_score(R_pred,test)
        top_10_precision_value = top_10_precision(R_pred,test)
        top_1_precision_value = top_1_precision(R_pred,test)
        precision_at_k_value = precision_at_k(R_pred,test)
        result = pd.DataFrame()
        result['RMSE']=[rmse_value]
        result['MAE']= [mae_value]
        result['top_10_f1_score']= [top_10_f1_score_value]
        result['top_10_precision']= [top_10_precision_value]
        result['top_1_precision']= [top_1_precision_value]
        result['precision_at_k']= [precision_at_k_value]
        print(result)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"k20_large_fixoutput_{timestamp}.csv"
        # result.to_csv(filename, index=False)


# In[36]:


df_onehot.values


# In[37]:



zxz = k_fold_cross_validation(rating_matrix.replace(to_replace=0, value=np.nan).values)
train_m=zxz[0][0]
test_m =zxz[0][1]
train_m


# In[38]:



values = [10000,  1000,1000000000000]
combinations = list(itertools.product(values, repeat=5))
combinations_arr = np.array(combinations)
selected_combinations = random.sample(combinations, 20)
print(len(combinations_arr))
model = NMF(n_components=20, init='random', random_state=0) #n_component = KNN values
U_matrix = model.fit_transform(np.nan_to_num(train_m,0))
V_matrix = model.components_
filter_D = model.fit_transform(df_onehot.values)
D_matrix = model.components_


# In[39]:


L_i=np.array(calculate_L(similarity_matrix))/10


# In[40]:


param_grid = {'alpha': [100,0.01],
              'lambdas_': selected_combinations,
              'R_': [train_m],
              'P_': [0],
              'U_':[U_matrix],
              'V_':[V_matrix],
              'Q_': [df_onehot.values],
              'E_': [0],
              'D_': [D_matrix],
              'C_u_': [C_umatrix],
              'C_i_': [C_imatrix],
              'L_U_': [0],
              'L_V_': [L_i],
              'b_u_': [user_bias.values],
              'b_v_': [item_bias.values]}
best_params = grid_search(MultiMF,param_grid, test_m)
alpha = best_params['alpha']
lambdas = best_params['lambdas_']
# object = MultiMF(train_m,0,df_onehot.values,U_matrix,V_matrix,0,D_matrix,C_umatrix,C_imatrix,0,L_i,lambdas,user_bias.values,item_bias.values,alpha)
# object.evaluation(test_m)

