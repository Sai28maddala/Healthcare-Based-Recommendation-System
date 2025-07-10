#!/usr/bin/env python
# coding: utf-8

# In[89]:


#Loading the datasets

import pandas as pd


df1 = pd.read_csv(r"C:\Users\palan\Desktop\DATASETS_FOR_CSP\disease_symptom.csv")
df2 = pd.read_csv(r"C:\Users\palan\Desktop\DATASETS_FOR_CSP\disease_medicine.csv")
df3 = pd.read_csv(r"C:\Users\palan\Desktop\DATASETS_FOR_CSP\disease_precautions.csv")
df4 = pd.read_csv(r"C:\Users\palan\Desktop\DATASETS_FOR_CSP\disease_diets.csv")


print("Dataset 1:")
display(df1.head())

print("\nDataset 2:")
display(df2.head())

print("\nDataset 3:")
display(df3.head())

print("\nDataset 4:")
display(df4.head())


# In[90]:


#PHASE-1:DATA PREPROCESSING

#STAGE-1: DATA CLEANING

#STEP-1: Handling Missing Values with the mode for each dataset

for df in [df1, df2, df3, df4]:
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)


# In[91]:


#After Handling Missing Values 

print("Dataset 1:")
display(df1.head())

print("\nDataset 2:")
display(df2.head())

print("\nDataset 3:")
display(df3.head())

print("\nDataset 4:")
display(df4.head())


# In[92]:


#MERGING DATASETS on common column-"Disease"

merged_df = pd.merge(df1, df2, on='Disease')
merged_df = pd.merge(merged_df, df3, on='Disease')
merged_df = pd.merge(merged_df, df4, on='Disease')




# In[93]:


#After Merging

merged_df.head()


# In[94]:


#STEP-2:REMOVING DUPLICATES (ALL DUPLICATES)-
'''
that is, removing exact duplicate records (same values in all columns) that do not add any unique information.
This can help reduce noise in your data.

'''
print("Shape before removing duplicates:", merged_df.shape)
merged_df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", merged_df.shape)


# In[95]:


print(list(merged_df.columns))


# In[96]:


# Apply one-hot encoding to all categorical columns
encoded_df = pd.get_dummies(merged_df, columns=['Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
                                                ' medicine_1', ' medicine_2', ' medicine_3', ' medicine_4',
                                                ' medicine_5', ' medicine_6', 'Precaution_1', 'Precaution_2',
                                                'Precaution_3', 'Precaution_4', 'Diet'], drop_first=True)

# View the first few rows of the encoded dataframe
encoded_df.head()


# In[97]:


print(list(encoded_df.columns))


# In[100]:


# List of symptom columns in the encoded_df
symptom_columns = [col for col in encoded_df.columns if col.startswith("Symptom")]

# Create a dictionary to hold the new columns
experienced_columns = {}

# Initialize each column with a list of zeros, matching the length of the original DataFrame
for symptom in symptom_columns:
    experienced_col_name = 'Experienced_' + symptom  # Create column name like 'Experienced_Symptom_1'
    experienced_columns[experienced_col_name] = [0] * len(encoded_df)  # Initialize all values as 0 (No)

# Convert the dictionary to a DataFrame
experienced_df = pd.DataFrame(experienced_columns)

# Concatenate the new DataFrame with the original one
encoded_df = pd.concat([encoded_df, experienced_df], axis=1)

# Check the updated dataframe to ensure the new columns are added
print(encoded_df.head())


# In[101]:


print(list(encoded_df.columns))


# In[107]:


# Extract column names that start with "Symptom" and "Experienced" and merge them into one list
symptom_columns_1 = [col for col in encoded_df.columns if col.startswith("Symptom")]
symptom_columns_2 = [col for col in encoded_df.columns if col.startswith("Experienced")]
symptom_columns=symptom_columns_1 + symptom_columns_2

X = encoded_df[symptom_columns]

# Extract column names that start with "Disease"
target_columns_1= [col for col in encoded_df.columns if col.startswith("Disease")]
target_columns_2= [col for col in encoded_df.columns if col.startswith("medicine")]
target_columns_3= [col for col in encoded_df.columns if col.startswith("Precaution")]
target_columns_4= [col for col in encoded_df.columns if col.startswith("Diet")]
target_columns= target_columns_1+target_columns_2+target_columns_3+target_columns_4

y = encoded_df[target_columns]


# In[108]:


from sklearn.model_selection import train_test_split

# First split: 70% training and 30% temp (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Second split: 20% validation and 10% test from the temp set
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
 
# Display shapes of the resulting datasets to confirm the split
print("Shape of training set (X_train):", X_train.shape)
print("Shape of validation set (X_val):", X_val.shape)
print("Shape of test set (X_test):", X_test.shape) 


# In[109]:


# Check for NaN values in X_train and y_train
print("Missing values in X_train before fitting:", X_train.isnull().sum().sum())
print("Missing values in y_train before fitting:", y_train.isnull().sum().sum())


# In[110]:


from sklearn.impute import SimpleImputer

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data
X_train_imputed = imputer.fit_transform(X_train)
# Transform the validation and test data
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Convert the imputed arrays back to DataFrames (if needed)
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_val.columns)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)


# Fit the imputer on the training data
y_train_imputed = imputer.fit_transform(y_train)
# Transform the validation and test data
y_val_imputed = imputer.transform(y_val)
y_test_imputed = imputer.transform(y_test)

# Convert the imputed arrays back to DataFrames (if needed)
y_train_imputed = pd.DataFrame(y_train_imputed, columns=y_train.columns)
y_val_imputed = pd.DataFrame(y_val_imputed, columns=y_val.columns)
y_test_imputed = pd.DataFrame(y_test_imputed, columns=y_test.columns)


# In[149]:


# Check for NaN values in X_train_imputed and y_train_imputed
print("Missing values in X_train_imputed after imputation:", X_train_imputed.isnull().sum().sum())
print("Missing values in y_train_imputed after imputation:", y_train_imputed.isnull().sum().sum())
print("Missing values in X_val_imputed after imputation:", X_val_imputed.isnull().sum().sum())
print("Missing values in y_val_imputed after imputation:", y_val_imputed.isnull().sum().sum())
print("Missing values in X_test_imputed after imputation:", X_test_imputed.isnull().sum().sum())
print("Missing values in y_test_imputed after imputation:", y_test_imputed.isnull().sum().sum())


# In[150]:


# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Initialize the Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust alpha for regularization strength

# Step 2: Fit the model on the training data
ridge_model.fit(X_train_imputed, y_train_imputed)


# In[151]:


# Step 3: Make predictions on the validation set
y_val_pred = ridge_model.predict(X_val_imputed)

# Step 4: Compute the loss (Mean Squared Error) and R^2 Score for validation set
mse = mean_squared_error(y_val_imputed, y_val_pred)
r2 = r2_score(y_val_imputed, y_val_pred)

print("Validation Mean Squared Error:", mse)
print("Validation R^2 Score:", r2)

# If you want to see the weights learned by the model
weights = ridge_model.coef_
print("\nModel Weights:", weights)


# In[153]:


# Step 2: Define the parameter grid to search over
from sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.1]  # You can modify this list to include more values
}

# Step 3: Set up the GridSearchCV
grid_search = GridSearchCV(estimator=ridge_model, 
                           param_grid=param_grid, 
                           scoring='neg_mean_squared_error',  # Use negative MSE because lower is better
                           cv=2)  # 5-fold cross-validation

# Step 4: Fit the model on the training data
grid_search.fit(X_train_imputed, y_train_imputed)

# Step 5: Get the best parameters and the corresponding score
best_alpha = grid_search.best_params_['alpha']
best_mse = -grid_search.best_score_  # Convert back to positive MSE

print("Best alpha:", best_alpha)
print("Best Mean Squared Error from Cross-Validation:", best_mse)



# In[155]:


# Step 6: Evaluate the best model on the validation set
best_model = grid_search.best_estimator_
y_val_pred_best = best_model.predict(X_val_imputed)

# Compute validation MSE and R^2 score for the best model
mse_best = mean_squared_error(y_val_imputed, y_val_pred_best)
r2_best = r2_score(y_val_imputed, y_val_pred_best)

print("Validation Mean Squared Error of Best Model:", mse_best)
print("Validation R^2 Score of Best Model:", r2_best)


# In[156]:


# Step 1: Create the Ridge model with the best alpha
best_alpha = 0.1  # Replace with your best alpha from GridSearchCV
ridge_model_best = Ridge(alpha=best_alpha)

# Step 2: Fit the model on the training set
ridge_model_best.fit(X_train_imputed, y_train_imputed)

# Step 3: Predict on the validation set
y_val_pred = ridge_model_best.predict(X_val_imputed)

# Step 4: Evaluate the model on the validation set
val_mse = mean_squared_error(y_val_imputed, y_val_pred)
val_r2 = r2_score(y_val_imputed, y_val_pred)

print("Validation Mean Squared Error:", val_mse)
print("Validation R^2 Score:", val_r2)


# In[157]:


from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(ridge_model_best, X_train_imputed, y_train_imputed, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive MSE
train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training MSE', color='blue')
plt.plot(train_sizes, val_scores_mean, label='Validation MSE', color='green')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[158]:


import joblib

# Save the model
joblib.dump(ridge_model_best, 'ridge_model_best.pkl')

print("Ridge Regression model saved successfully!")


# In[159]:


import os

# Check the current working directory
current_directory = os.getcwd()
print("The model is saved in:", current_directory)

# List files in the current directory
files = os.listdir()
print("Files in the current directory:", files)


# In[ ]:





# In[ ]:




