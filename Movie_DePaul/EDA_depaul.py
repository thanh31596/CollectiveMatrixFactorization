#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# from stellargraph import StellarGraph
# from stellargraph.mapper import At
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
# from stellargraph.layer import Attri2Vec
# from tensorflow.keras import Model, optimizers, losses


# In[2]:


df = pd.read_csv('Movie_DePaulMovie/ratings.txt',sep=',')


# In[3]:


from itertools import product

def grid_search(model, param_grid, n_iter=300):
    best_loss = float('inf')
    best_params = None
    for params in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), params))
        mf = model(**params)
        mf.factorize(iter=n_iter)
        loss = mf.total_loss[-1]
        if loss < best_loss:
            best_loss = loss
            best_params = params
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

from sklearn.impute import KNNImputer


# In[12]:


location = df[['userid','Location']]


# In[13]:


col_sub=location['Location'].unique()


# In[14]:


group_location = pd.get_dummies(location).groupby('userid').max().reset_index()


# In[15]:


df = pd.read_csv('Movie_DePaulMovie/ratings.txt',sep=',')
print(df.columns)
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


# In[ ]:





# In[18]:


neighbors=int(df['userid'].value_counts().mean())
imputer = KNNImputer(n_neighbors=neighbors)
imputed_df = pd.DataFrame(imputer.fit_transform(df),columns=['userid', 'itemid', 'rating', 'Time', 'Location', 'Companion'])

for i in imputed_df.columns: 
    imputed_df[i] = imputed_df[i].astype('int')


# In[19]:


imputed_df.info()


# In[ ]:





# # Generating Contextual Coeficient

# In[20]:


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


# In[21]:


imputed_df['Time_user'] = df_replaced['Time']
imputed_df['Location_user']= df_replaced['Location']
imputed_df['Companion_user']= df_replaced['Companion']
imputed_df['Time_item'] = df_replaced_['Time']
imputed_df['Location_item']= df_replaced_['Location']
imputed_df['Companion_item']= df_replaced_['Companion']
imputed_df


# # Get Bias

# In[22]:


df_replaced.head(30)
global_mean = df_replaced["rating"].mean()
user_bias = df_replaced.groupby("userid")["rating"].mean() - global_mean
item_bias = df_replaced.groupby("itemid")["rating"].mean() - global_mean
df_replaced['userid']


# In[23]:


from sklearn.decomposition import NMF

rating_matrix = df_replaced[['userid','itemid','rating']].pivot_table(values='rating',index='userid',columns='itemid')
model = NMF(n_components=neighbors, init='random', random_state=0) #n_component = KNN values
U_matrix = model.fit_transform(rating_matrix.fillna(0).values)
V_matrix = model.components_


# In[24]:



def to_int(x):
    if pd.isna(x):
        return x
    return int(x)
rating_matrix=rating_matrix.applymap(to_int)
def objective_function(R, P, Q, U, V, E, D, C_u,C_i,  L_U, L_V, lambdas, b_u, b_v):
    """Objective function

    Args:
        R (scalar): rating value
        P (matrix): user attribute matrix (uxp)
        Q (matrix): item attribute matrix (vxq)
        U (vector): user latent vector
        V (vector): item latent vector
        E (matrix): random matrix shape pxk
        D (maitr): random matrix shape qxk
        C (scalar): contextual coefficient
        L_U (matrix): laplacian matrix for U
        L_V (matrix): laplacian matrix for V
        lambdas (list): a list of lambdas
        b_u (scalar): bias of user
        b_v (scalar): bias of item

    Returns:
        loss: loss value 
    """
    #First situation: both user and item attributes are not available
    if P == None and Q ==None and E == None and D == None: 
        P =0
        E =0
        D = 0 
        Q = 0
    U=U+0.000000000005
    V=V+0.000000000005
    M = U.shape[0] #number of user
    V_=0
    U_=0
    for ue,ve in zip(C_u,C_i):
        U_ += ue*U
        V_ += ve*V
    U_mean=U_/M
    V_mean=V_/M
    R_pred = (U_mean + U) @ (V.T + V_mean)
    if P !=0:
        P_pred = U @ E.T
        Q_pred = V @ D.T
    else:
        P_pred = 0
        Q_pred = 0
    
    loss = np.sum((R - R_pred)**2) + np.sum((P - P_pred)**2) + np.sum((Q - Q_pred)**2)
    loss += lambdas[0] * np.sum(U**2) + lambdas[1] * np.sum(V**2) + lambdas[2] * np.sum(D**2) + lambdas[3] * np.sum(E**2)
    # loss += lambdas[4] * np.trace(U.T @ L_U @ U) + lambdas[5] * np.trace(V.T @ L_V @ V)
    loss += b_u + b_v
    
    return loss


# # Get Laplacian

# In[25]:


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



from tqdm import tqdm
# Compute adjacency matrices based on similarity
user_adj_matrix = adjacency_matrix_similarity(U_matrix)
item_adj_matrix = adjacency_matrix_similarity(V_matrix)

# Compute Laplacian matrices
L_U = laplacian_matrix(user_adj_matrix)
L_V = laplacian_matrix(item_adj_matrix)
# Get context matrix
scaler = StandardScaler()

Cu=imputed_df[['Time_user',	'Location_user',	'Companion_user']]
Ci=imputed_df[['Time_item',	'Location_item',	'Companion_item']]
C_umatrix =  scaler.fit_transform(Cu)
C_imatrix =  scaler.fit_transform(Ci)


# In[26]:


import numpy as np
from sklearn.decomposition import NMF
class MultiMF: 
    def __init__(self,R_, P_, Q_, U_, V_, E_, D_, C_u_,C_i_,  L_U_, L_V_, lambdas_, b_u_, b_v_,alpha):
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
            if abs(self.total_loss[-1] - self.total_loss[-2]) < 0.00000004:
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
        return self.U_matrix, self.V_matrix
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

                    U, V, D, E = self.update_U_V_E_D(R=R, P=0, E=0, D=0, Q=0, U=U, V=V, C_u=C_u, C_i=C_i, lambdas=lambdas)
                    loss,U_o,V_o,D_o,E_o = self.objective_function(R, P, Q, U, V, E, D, C_u,C_i,  L_U, L_V, lambdas, b_u, b_v)

                    self.U_matrix[i] = U
                    self.V_matrix.T[j] = V
                    
                    loss_row+=loss
                    # U_ = self.update_U(R, P, Q, U, V, E, D, C_u, C_i, lambdas, alpha)
            loss_col+=loss_row
        self.total_loss.append(loss_col)
    def objective_function(self,R, P, Q, U, V, E, D, C_u,C_i,  L_U, L_V, lambdas, b_u, b_v):
        """Objective function

        Args:
            R (scalar): rating value
            P (matrix): user attribute matrix (uxp)
            Q (matrix): item attribute matrix (vxq)
            U (vector): user latent vector
            V (vector): item latent vector
            E (matrix): random matrix shape pxk
            D (maitr): random matrix shape qxk
            C (scalar): contextual coefficient
            L_U (matrix): laplacian matrix for U
            L_V (matrix): laplacian matrix for V
            lambdas (list): a list of lambdas
            b_u (scalar): bias of user
            b_v (scalar): bias of item

        Returns:
            loss: loss value 
        """
        #First situation: both user and item attributes are not available

        U=U+0.000000000005
        V=V+0.000000000005
        M = U.shape[0] #number of user
        V_=0
        U_=0
        for ue,ve in zip(C_u,C_i):
            U_ += ue*U
            V_ += ve*V
        U_mean=U_/M
        V_mean=V_/M
        R_pred = (U_mean + U) @ (V.T + V_mean)
        if P !=0:
            P_pred = U @ E.T
            Q_pred = V @ D.T
        else:
            P_pred = 0
            Q_pred = 0
        
        loss = np.sum((R - R_pred)**2) + np.sum((P - P_pred)**2) + np.sum((Q - Q_pred)**2)
        loss += lambdas[0] * np.sum(U**2) + lambdas[1] * np.sum(V**2) + lambdas[2] * np.sum(D**2) + lambdas[3] * np.sum(E**2)
        # loss += lambdas[4] * np.trace(U.T @ L_U @ U) + lambdas[5] * np.trace(V.T @ L_V @ V)
        loss += b_u + b_v
        
        return loss,U,V,D,E

    def update_U_V_E_D(self,R, P, Q, U, V, E, D, C_u,C_i,lambdas):
        alpha=self.alpha
        U=U+0.000000000005
        V=V+0.000000000005
        M = U.shape[0] #number of user
        V_=0
        U_=0
        for ue,ve in zip(C_u,C_i):
            U_ += ue*U
            V_ += ve*V
        U_mean=U_/M
        V_mean=V_/M
        if P !=0:
            P_pred = U @ E.T
            Q_pred = V @ D.T
        else:
            P_pred = 0
            Q_pred = 0
        R_pred = (U_mean + U) @ (V.T + V_mean)


        dU = -2 * (R- R_pred) * (V.T + V_mean) - 2 * (P - P_pred) * E + 2 * lambdas[0] * U   
        dV = -2 * (R - R_pred) * (U_mean + U) - 2 * (Q - Q_pred) * D + 2 * lambdas[1] * V  
        dE = -2 * (P - P_pred) * U + 2 * lambdas[3] * E
        dD = -2 * (Q - Q_pred) * V + 2 * lambdas[2] * D

        U -= alpha * dU
        V -= alpha * dV
        E -= alpha * dE
        D -= alpha * dD

        return U, V, E, D


# In[27]:


param_grid = {'alpha': [0.001, 0.01, 0.1],
              'lambdas_': [(0.01, 0.01, 0.01, 0.01), 
                           (0.1, 0.1, 0.1, 0.1), 
                           (1, 1, 1, 1),(0.3, 0.3, 0.3, 0.3)],
              'R_': [rating_matrix.values],
              'P_': [0],
              'Q_': [0],
              'U_': [U_matrix],
              'V_': [V_matrix],
              'E_': [0],
              'D_': [0],
              'C_u_': [C_umatrix],
              'C_i_': [C_imatrix],
              'L_U_': [0],
              'L_V_': [0],
              'b_u_': [user_bias.values],
              'b_v_': [item_bias.values]}
best_params = grid_search(MultiMF, param_grid)
alpha = best_params['alpha']
lambdas = best_params['lambdas_']
object = MultiMF(rating_matrix.values,0,0,U_matrix,V_matrix,0,0,C_umatrix,C_imatrix,0,0,lambdas,user_bias.values,item_bias.values,alpha)


# In[28]:


U_result, V_result=object.factorize(300)


# In[29]:


import matplotlib.pyplot as plt
plt.plot(object.total_loss[1:])


# In[55]:


rating_matrix


# In[30]:


from sklearn.preprocessing import MinMaxScaler
mmscaler=MinMaxScaler(feature_range=(1, 6))


# In[31]:



R_pred=np.dot(U_result,V_result)
R_pred=mmscaler.fit_transform(R_pred).astype(int)


# In[32]:


def top_10_precision(predicted_ratings, real_ratings):
    precision_sum = 0
    users_count = real_ratings.shape[0]

    for user_idx in range(users_count):
        user_real_ratings = real_ratings.values[user_idx]
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
        user_real_ratings = real_ratings.values[user_idx]
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


# In[33]:


top_10_precision(R_pred,rating_matrix)
top_1_precision(R_pred,rating_matrix)


# In[34]:


top_1_precision(R_pred,rating_matrix)


# In[40]:


mod = NMF(n_components=52, init='random', random_state=0)
z = model.fit_transform(X)
c = model.components_


# In[41]:


R_pred_benchmark1 = np.dot(z,c)
R_pred_benchmark1=mmscaler.fit_transform(R_pred_benchmark1).astype(int)


# In[42]:


print(top_10_precision(R_pred_benchmark1,rating_matrix))
print(top_1_precision(R_pred_benchmark1,rating_matrix))


# In[52]:


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


# In[54]:


print("Top 10: ",top_10_precision(R_pred,rating_matrix))
print("Top 1 Precision: ",top_1_precision(R_pred,rating_matrix))
print("RMSE: ",rmse(R_pred,rating_matrix.values))
print("MAE: ",mae(R_pred,rating_matrix.values))


# In[50]:


result = pd.DataFrame()
result['Top10']= top_10_precision(R_pred,rating_matrix)
result['Top1']= top_1_precision(R_pred,rating_matrix)
result['RMSE']= top_10_precision(R_pred,rating_matrix)
result['alpha']= best_params['alpha']
result['lambdas_']= best_params['lambdas_']
result.to_csv('result.csv')


# In[48]:


best_params['alpha']
best_params['lambdas_']


# In[ ]:




