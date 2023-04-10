#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

warnings.filterwarnings("ignore")
np.random.seed(42)
scaler_minmax=MinMaxScaler()

# In[2]:


df = pd.read_csv('depaul_full.csv')
origin_data = pd.read_csv('ratings.txt',sep=',')
origin_data=origin_data.fillna(0)


# In[3]:


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


# def grid_search(model, param_grid, n_iter=200, filename='grid_search_results.xlsx'):
#     best_loss = float('inf')
#     best_params = None
#     results = []
#     for params in product(*param_grid.values()):
#         params = dict(zip(param_grid.keys(), params))
#         mf = model(**params)
#         mf.factorize(iter=n_iter)
#         loss = mf.total_loss[-1]
#         results.append({**params, 'loss': loss})
#         if loss < best_loss:
#             best_loss = loss
#             best_params = params
#     print('Best parameters:', best_params)
#     print('Best loss:', best_loss)
#     df = pd.DataFrame(results)
#     df.to_excel(filename, index=False)
#     return best_params

def factorize(model, params, n_iter):
    mf = model(**params)
    mf.factorize(iter=n_iter)
    loss = mf.total_loss[-1]
    return {**params, 'loss': loss}

def grid_search(model, param_grid, n_iter=10, n_jobs=2, filename='grid_search_results.xlsx'):
    best_loss = float('inf')
    best_params = None
    results = []

    with Pool(n_jobs) as pool:
        params_list = [dict(zip(param_grid.keys(), p)) for p in product(*param_grid.values())]
        factorize_args = [(model, p, n_iter) for p in params_list]
        results = pool.starmap(factorize, factorize_args)

    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)

    best_params = df.loc[df['loss'].idxmin()].to_dict()
    best_loss = best_params.pop('loss')

    print('Best parameters:', best_params)
    print('Best loss:', best_loss)

    return best_params


# In[4]:


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


# In[5]:


def count_nan(df):
    """
    Returns the percentage of NaN values in a pandas DataFrame.
    """
    total_cells = df.size
    nan_cells = df.isna().sum().sum()
    nan_percentage = (nan_cells / total_cells) * 100
    print(nan_percentage)
count_nan(df)


# In[6]:


df['rating'].value_counts()


# In[7]:


# neighbors=int(df['userid'].value_counts().mean())
# imputer = KNNImputer(n_neighbors=neighbors)
# imputed_df = pd.DataFrame(imputer.fit_transform(df),columns=['userid', 'itemid', 'rating', 'Time', 'Location', 'Companion'])

# for i in imputed_df.columns: 
#     imputed_df[i] = imputed_df[i].astype('int')
# imputed_df


# In[8]:


# imputed_df.info()


# In[ ]:





# # Generating Contextual Coeficient

# In[9]:




# Assuming your dataframe is called df
X = df[['Time', 'Location', 'Companion','userid']]
y = df['rating']

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

X = df[['Time', 'Location', 'Companion','itemid']]
y = df['rating']

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
features = ['Time', 'Location', 'Companion','itemid']
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


# In[10]:


df_fix_user = df_replaced[['Time','Location','Companion']]
for i in ['Time','Location','Companion']:
    for j in range(len(df_fix_user[i])):
        if origin_data[i][j] ==0: 
            df_replaced[i][j]=0
df_fix_user = df_replaced_item[['Time','Location','Companion']]
for i in ['Time','Location','Companion']:
    for j in range(len(df_fix_user[i])):
        if origin_data[i][j] ==0: 
            df_replaced_item[i][j]=0      
df['Time_user'] = df_replaced['Time']
df['Location_user']= df_replaced['Location']
df['Companion_user']= df_replaced['Companion']
df['Time_item'] = df_replaced_item['Time']
df['Location_item']= df_replaced_item['Location']
df['Companion_item']= df_replaced_item['Companion']
try:
    df=df.drop(columns=['Unnamed: 0'])
except KeyError:
    pass
df.describe()


# In[11]:


df


# # Get Bias

# In[12]:


df_replaced.head(30)
global_mean = df_replaced["rating"].mean()
user_bias = df_replaced.groupby("userid")["rating"].mean() - global_mean
item_bias = df_replaced.groupby("itemid")["rating"].mean() - global_mean
print(np.min(user_bias))
print(np.max(user_bias))
print(np.mean(user_bias))
print("SUM: ",np.sum(user_bias))
print(np.min(item_bias))
print(np.max(item_bias))
print(np.mean(item_bias))
print("SUM: ",np.sum(item_bias))


# In[13]:



rating_matrix_original= df[['userid','itemid','rating']].pivot_table(values='rating',index='userid',columns='itemid', fill_value=0).astype('int')
rating_matrix_original.values


# In[14]:



def to_int(x):
    if pd.isna(x):
        return x
    return int(x)
rating_matrix=rating_matrix_original
rating_matrix


# # Get Laplacian

# In[15]:


def adjacency_matrix_similarity(matrix):
    similarity_matrix = matrix @ matrix.T
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix

def degree_matrix(adj_matrix):
    degree_vector = np.sum(adj_matrix, axis=1)
    return np.diag(degree_vector)

def laplacian_matrix(adj_matrix):
    deg_matrix = degree_matrix(adj_matrix)
    return deg_matrix - adj_matrix
neighbors=52

from tqdm import tqdm
# Compute adjacency matrices based on similarity
# user_adj_matrix = adjacency_matrix_similarity(U_matrix)
# item_adj_matrix = adjacency_matrix_similarity(V_matrix)
# # Compute Laplacian matrices
# L_U = laplacian_matrix(user_adj_matrix)
# L_V = laplacian_matrix(item_adj_matrix)
# Get context matrix
scaler = StandardScaler()
# minmax = MinMaxScaler(feature_range=(0,2))
Cu=df[['Time_user',	'Location_user',	'Companion_user']].multiply(100000)
Ci=df[['Time_item',	'Location_item',	'Companion_item']].multiply(100000)
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


# In[16]:


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


# In[17]:




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
    def __init__(self,R_, P_, Q_,U_,V_,  E_, D_, C_u_,C_i_,  L_U_, L_V_, lambdas_, b_u_, b_v_,alpha):

        self.R_matrix = np.array(R_)
        self.P_matrix = np.array(P_) 
        self.Q_matrix = np.array(Q_) 
        self.lambdas = lambdas_ 
        self.U_matrix = np.array(U_)
        self.V_matrix = np.array(V_)
        self.E_matrix = np.array(E_)
        self.D_matrix = np.array(D_) 
        self.C_umatrix=np.array(C_u_) 
        self.C_imatrix=np.array(C_i_) 
        self.L_U=L_U_
        self.L_V=L_V_
        self.item_bias=np.array(b_v_)
        self.user_bias=np.array(b_u_)
        self.newU=[]
        self.newV=[]
        self.newE=[]
        self.newD=[]
        self.alpha=alpha
        self.total_loss=[0]
    def factorize(self,iter=10):
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
        loss_col = 0 
        lambdas=self.lambdas
        for i in range(len(self.R_matrix)):  #70
            loss_row = 0
            U = self.U_matrix[i]
            C_u = self.C_umatrix[i,:]
            C_i = self.C_imatrix[i,:]
            b_u=self.user_bias[i]
            for j in range(len(self.R_matrix[i])): #97

                if not np.isnan(self.R_matrix[i,j]):

                    R = self.R_matrix[i,j]
                    V = self.V_matrix.T[j]
                    b_v=self.item_bias[j]
                    #First situation
                    P=Q=L_U=L_V=E=D=0

                    U, V, D, E = self.update_U_V_E_D(R=R, P=0, E=0, D=0, Q=0, U=U, V=V, C_u=C_u, C_i=C_i,Lu=0,Li=0, lambdas=lambdas)
                    self.U_matrix[i] = U
                    self.V_matrix.T[j] = V
                    loss,U_o,V_o,D_o,E_o = self.objective_function(R, P, Q, U, V, E, D, C_u,C_i,  L_U, L_V, lambdas, b_u, b_v)
                    assert not (np.isnan(U).any() or np.isnan(V).any())


                    loss_row+=loss
                    # U_ = self.update_U(R, P, Q, U, V, E, D, C_u, C_i, lambdas, alpha)
            loss_col+=loss_row
        self.total_loss.append(loss_col)
    def objective_function(self,R, P, Q, U, V, E, D, C_u,C_i,  L_U, L_V, lambdas, b_u, b_v):
        # U=U+0.0000000000005
        # V=V+0.0000000000005
        M = U.shape[0] #number of user
        V_=0
        U_=0
        
        U_ += np.sum(C_u)*U
        V_ += np.sum(C_i)*V
        U_mean=U_/M
        V_mean=V_/M
        R_pred = np.dot(U_mean + U, V.T + V_mean)
        if P !=0:
            P_pred = np.dot(U, E.T)
            Q_pred = np.dot(V, D.T)
        else:
            P_pred = 0
            Q_pred = 0
        loss = np.sum((R - R_pred)**2) + np.sum((P - P_pred)**2) + np.sum((Q - Q_pred)**2)
        loss += lambdas[0] * np.sum(U**2) + lambdas[1] * np.sum(V**2) + lambdas[2] * np.sum(D**2) + lambdas[3] * np.sum(E**2)
        loss += b_u + b_v
        
        return loss,U,V,D,E

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
        if P != 0:
            P_pred = np.dot(U, E.T)
            Q_pred = np.dot(V, D.T)
        else:
            P_pred = 0
            Q_pred = 0
        R_pred = np.dot((U_mean + U), (V.T + V_mean))


        dU = -2 * (R- R_pred) * (V.T + V_mean)*(1+U_mean) - 2 * (P - P_pred) * E + 2 * lambdas[0] * U   + 2*lambdas[3]*Lu #change to lambda 5 soon
        dV = -2 * (R - R_pred) * (U_mean + U)*(1+V_mean) - 2 * (Q - Q_pred) * D + 2 * lambdas[1] * V  + 2*lambdas[3]*Li
        dE = -2 * (P - P_pred) * U + 2 * lambdas[3] * E
        dD = -2 * (Q - Q_pred) * V + 2 * lambdas[2] * D
        
        U -= alpha * dU
        V -= alpha * dV
        E -= alpha * dE
        D -= alpha * dD
        Uz= scaler_minmax.fit_transform(U.reshape(-1, 1))
        Vz= scaler_minmax.fit_transform(V.reshape(-1, 1))
        Ez= scaler_minmax.fit_transform(E.reshape(-1, 1))
        Dz= scaler_minmax.fit_transform(D.reshape(-1, 1))

        return Uz.T[0],Vz.T[0],Ez.T[0],Dz.T[0]
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
        filename = f"output_{timestamp}.csv"
        result.to_csv(filename, index=False)


zxz = k_fold_cross_validation(rating_matrix.replace(to_replace=0, value=np.nan).values)
train_m=zxz[0][0]
test_m =zxz[0][1]

values = [0.001, 0.01, 0.1, 0, 1, 10, 100]
values = [0.001,  0.1]
combinations = list(itertools.product(values, repeat=4))
combinations_arr = np.array(combinations)
len(combinations_arr)
model = NMF(n_components=10, init='random', random_state=0) #n_component = KNN values
U_matrix = model.fit_transform(np.nan_to_num(train_m,0))
V_matrix = model.components_
# U_matrix[U_matrix == 0] = 0.000005
# V_matrix[V_matrix == 0] = 0.000005


# In[21]:


param_grid = {'alpha': [1,0.01],
              'lambdas_': combinations_arr,
              'R_': [train_m],
              'P_': [0],
              'U_':[U_matrix],
              'V_':[V_matrix],
              'Q_': [0],
              'E_': [0],
              'D_': [0],
              'C_u_': [C_umatrix],
              'C_i_': [C_imatrix],
              'L_U_': [0],
              'L_V_': [0],
              'b_u_': [user_bias.values],
              'b_v_': [item_bias.values]}
best_params = grid_search(MultiMF, param_grid)
# alpha = best_params['alpha']
# lambdas = best_params['lambdas_']
# k = best_params['k']
# object = MultiMF(rating_matrix.values,0,0,0,0,C_umatrix,C_imatrix,0,0,lambdas,user_bias.values,item_bias.values,alpha)
# U_result, V_result=object.factorize(5)


# In[22]:


object = MultiMF(train_m,0,0,U_matrix,V_matrix,0,0,C_umatrix,C_imatrix,0,0,[0.1,0.01,0.1,0.001],user_bias.values,item_bias.values,0.001)
# object.factorize(5)
object.evaluation(test_m)


# In[23]:


# import matplotlib.pyplot as plt
# plt.plot(object.total_loss[1:])

