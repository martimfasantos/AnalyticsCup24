#!/usr/bin/env python
# coding: utf-8

# ## <font style="font-weight: bold;"> Analytics Cup 2024 </font>

# In[4]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Note: The following do not work with Python 3.12
import shap
from ydata_profiling import ProfileReport
import sweetviz as sv

import os
from matplotlib.pyplot import figure, title, savefig, show, tight_layout, subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from pandas import concat, DataFrame
from sklearn.impute import SimpleImputer
from numpy import nan
from fractions import Fraction
from seaborn import heatmap
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score


# #### Reproducibility

# In[5]:


seed = 2024

# pandas, statsmodels, matplotlib and y_data_profiling rely on numpy's random generator, and thus, we need to set the seed in numpy
np.random.seed(seed)


# ### <font color='green'> Phase 1: Business Understanding </font>
# 
# Business Understanding is the first and economically most important step in the
# CRISP-DM process. It serves to assess use cases, feasibility, requirements, and
# risks of the endeavored data driven project. Since the conduction of data driven
# projects usually depends on the data at hand, the CRISP-DM process often 
# alternates between Business Understanding and Data Understanding, until the
# project's schedule becomes sufficiently clear.
# 
# #### Business Understanding
# 
# In LLMeals Analytics Cup, the goal is to improve customer satisfaction and reduce subscription cancellations by developing a model that accurately predicts whether a customer likes (Like=1) or dislikes (Like=0) a suggested recipe. The model will utilize datasets such as "recipes.csv," "reviews.csv," "diet.csv," and "requests.csv" to generate insights into user preferences. The successful model will serve as the foundation for enhancing the quality of suggested recipes in LLMeals' service, aligning them more closely with individual customer requirements. The project's ultimate aim is to leverage data-driven approaches for refining the recipe suggestions and, in turn, improving the overall LLMeals user experience.

# ### <font color='green'> Phase 2: Data Understanding </font>
# 
# The *Data Understanding* phase mainly serves to inform the Business Understanding step by
# assessing the data quality and content, and should provide the engineers with 
# an intuition for the specific data and the specific problem at hand. Experienced
# data scientists and machine learning engineers can often estimate the difficulty
# and feasibility of the task by analyzing and understanding the data.  
# 
# #### Example: Data Understanding
# 
# Make yourself familiar with the structure and content of the data. *Note*, this step 
# heavily depends on the specific problem at hand, since there is no fixed recipe that 
# fits all possible data sets. In the example below, we are only looking at a very small
# data set and do **not** conduct an in-depth analysis.  

# In[6]:


# load the data
file_dir = "."
file_names = ["reviews.csv", "requests.csv", "diet.csv", "recipes.csv"]
reviews = pd.read_csv(f'{file_dir}/{file_names[0]}', low_memory=False)
requests = pd.read_csv(f'{file_dir}/{file_names[1]}')
diet = pd.read_csv(f'{file_dir}/{file_names[2]}')
recipes = pd.read_csv(f'{file_dir}/{file_names[3]}')


# Reviews

# In[7]:


# print(reviews.sample(3))
# print("\n")
# print(reviews.info())
# print("\n")
# print(reviews.describe())
# sns.boxplot(reviews)


# Requests

# In[8]:


# print(requests.sample(3))
# print("\n")
# print(requests.info())
# print("\n")
# print(requests.describe())
# sns.boxplot(requests)


# Diet

# In[9]:


# print(diet.sample(3))
# print("\n")
# print(diet.info())
# print("\n")
# print(diet.describe())
# sns.boxplot(diet)


# Recipes

# In[10]:


# print(recipes.sample(3))
# print("\n")
# print(recipes.info())
# print("\n")
# print(recipes.describe())
# plt.figure(figsize=(24, 6))
# sns.boxplot(recipes)


# #### Class Balance

# In[11]:


# # check the balancing of classes/labels
# print(reviews.groupby("Like").size())

# # -> 2 classes, 1 is much more frequent than the other (False/True ratio is ~4:1)


# #### Reviews Feature Distribution

# In[12]:


# have a look at the feature distributions with a pairplot,
# as it gives you a good overview over possible outliers
# and a good overview over the data in general

# pairplot for the full data
# columns_to_drop = ["AuthorId", "RecipeId", "TestSetId"]
# sns.pairplot(reviews.drop(columns_to_drop, axis=1), hue="Like", diag_kind="hist", diag_kws={"multiple" : "stack"})


# #### Requests Feature Distribution

# In[13]:


# have a look at the feature distributions with a pairplot,
# as it gives you a good overview over possible outliers
# and a good overview over the data in general

# pairplot for the full data
# columns_to_drop = ["AuthorId", "RecipeId"]
# data = pd.merge(requests, reviews[["AuthorId", "RecipeId", "Like"]], on=['AuthorId','RecipeId'], \
#     how='left').drop(columns_to_drop, axis=1)
# sns.pairplot(data, hue="Like", diag_kind="hist", diag_kws={"multiple" : "stack"})


# #### Diet Feature Distribution

# In[14]:


# pairplot for the full data
# columns_to_drop = ["AuthorId"]
# data = pd.merge(diet, reviews[["AuthorId", "Like"]], on=['AuthorId'], how='left').drop(columns_to_drop, axis=1)
# sns.pairplot(data, hue="Like", diag_kind="hist", diag_kws={"multiple" : "stack"})


# #### Recipes Feature Distribution

# In[15]:


# pairplot for the full data
# columns_to_drop = ["RecipeId"]
# data = pd.merge(recipes, reviews[["RecipeId", "Like"]], on=['RecipeId'], how='left').drop(columns_to_drop, axis=1)
# sns.pairplot(data, hue="Like", diag_kind="hist", diag_kws={"multiple" : "stack"})


# ### Create a merged dataset

# | **File**     | **Join Keys**            |
# |--------------|------------------------|
# | reviews.csv  | AuthorId, RecipeId     |
# | requests.csv | AuthorId, RecipeId     |
# | diet.csv     | AuthorId               |
# | recipes.csv  | RecipeId               |
# 

# In[16]:


# Merge the dataframes using multiple columns
merged_df = pd.merge(reviews, requests, on=['AuthorId','RecipeId'], how='left')
merged_df = pd.merge(merged_df, diet, on='AuthorId', how='left')
merged_df = pd.merge(merged_df, recipes, on='RecipeId', how='left')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_data.csv', index=True)

# Check if the number of rows and columns are correct
# print(len(merged_df) == len(reviews))
# print(reviews.shape)
# print(merged_df.shape)
# print(merged_df.shape[1] == reviews.shape[1] + requests.shape[1]-2 + diet.shape[1]-1 + recipes.shape[1]-1)


# ### Import merged dataset

# In[17]:


# load the data
file_dir = "."
file_name = "merged_data.csv"
df = pd.read_csv(f'{file_dir}/{file_name}', low_memory=False)


# #### Class-dependent pairplots

# ### Summary Report

# In[18]:


# # We can also leverage the dataprep package to get a nice summary report
# report = sv.analyze(df)
# report.show_notebook()

# # We can also leverage the yadata_profiling package to get a nice summary report
# profile = ProfileReport(df, title="LLMeals - Summary Report")
# profile


# #### Summary: Data Understanding
# 
# You should have a good understanding what the data is about and of some of its properties. Newly gained insights are used to reiterate the
# Business Understanding Phase, but in this example, it won't be necessary.

# ### <font color='green'> Phase 3: Data Preparation </font>
# 
# Data Preparation mainly consists of two parts, Data Cleaning and Data Wrangling. In Data
# Cleaning, the goal is assure data quality. This includes removing wrong/corrupt 
# data entries and making sure the entries are standardized, e.g. enforcing certain encodings. 
# Data Wrangling then transforms the data in order to make it suitable for the modelling step.
# Sometimes, steps from Data Wrangling are incorporated into the automatized Pipeline, as
# we will show in this example.

# ### Data Cleaning

# #### Variable Encoding

# In[19]:


file_source_path = 'merged_data.csv' # source file
file_dir = '.' # destination directory
file_tag = 'dataset'

df = pd.read_csv(file_source_path, low_memory=False)
index_column = df.columns[0]
df.drop([index_column], axis=1, inplace=True)

# ----------- Convertions ----------- #
# Like: object -> bool
# HighProtein: {Indiferent, Yes} - object -> bool
# LowSugar: {0, Indiferent} - object -> bool
# Diet: {Vegetarian, Omnivore, Vegan} - object -> categorical
# Name: 140k values, 50% distinct - object -> categorical
# RecipeCategory: 7 categories - object -> categorical
# RecipeIngredientQuantities: filter
# RecipeIngredientParts: filter
# RecipeYield: 46k values, 7.9k distinct, 93.8k missing - object -> categorical

# AUX: dummify the variables
def dummify_var(df, var_to_dummify):
    other_vars = [c for c in df.columns if not c in var_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int)
    X = df[var_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names_out(var_to_dummify)
    trans_X = encoder.transform(X)
    dummy = pd.DataFrame(trans_X, columns=new_vars, index=X.index)
    #dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = pd.concat([df[other_vars], dummy], axis=1)
    return final_df

# AUX: sum the numeric values
def sum_numeric_values(strings):
    cleaned_numbers = parse_ingredients(strings)    
    total_sum = 0
    try:
        for num in cleaned_numbers:
            nums = num.split('-')
            nums = [n.strip() for n in nums]
            quantities = []
            for n in nums:
                parts = n.split(' ')
                if len(parts) == 2:
                    # Handle mixed numbers
                    quantities.append(float(Fraction(parts[0])) + float(Fraction(parts[1])))
                    # print(f"{float(Fraction(parts[0]))} {float(Fraction(parts[1]))}")
                elif '/' in n:
                    # Handle fractions
                    quantities.append(float(Fraction(n)))
                    # print(Fraction(n))
                else:
                    # Handle regular floats
                    quantities.append(float(n))
                    # print(float(n))
            total_sum += np.mean(quantities)
    # Handle wrong format of data
    except:
        return 0
        
    return total_sum


# AUX: parse the ingredient parts
def parse_ingredients(ingredient_parts):    
    # Clean the ingredient parts
    cleaned_string = ingredient_parts.replace('c(', '').replace('"', '').replace('\\', '').replace(')', '').split(',')
    cleaned_ingredients = [ingredient.strip() for ingredient in cleaned_string]
    for ing in cleaned_ingredients:
        if 'character' in ing:
            cleaned_ingredients.remove(ing)
    return cleaned_ingredients

meat_keywords = ['chicken', 'beef', 'pork', 'shrimp', 'salmon', 'sausage', 'bacon', 'turkey', 'ham', 'lamb', \
    'fish', 'meatballs', 'steak', 'tuna', 'ground beef', 'venison', 'duck', 'lobster', 'crab', 'oysters', 'clams', \
    'mussels', 'scallops', 'squid', 'octopus', 'escargot', 'prawns', 'crawfish', 'bison', 'rabbit', 'quail', 'goose', \
    'foie gras', 'veal', 'liver', 'tripe', 'anchovies', 'sardines', 'haddock', 'halibut', 'catfish', 'swordfish', 'hamburger patties', \
    'hot dogs', 'corned beef', 'lamb chops', 'liverwurst', 'chorizo', 'pepperoni', 'salami', 'pastrami', 'prosciutto', 'pate']
vegetarian_keywords = ['butter', 'eggs', 'milk', 'cheese', 'cream', 'honey', 'mayonnaise', 'cheddar', \
    'cream', 'margarine', 'mustard', 'buttermilk', 'mozzarella', 'oil', 'jack', 'feta', 'yogurt', 'cheddar', \
    'cottage', 'provolone', 'gruyere', 'brie', 'romano', 'goat', 'ricotta', 'asiago', 'fontina', 'colby',  \
    'gouda', 'fraiche', 'gelatin']

def create_products_columns(row):
    name_keywords_meat = [keyword for keyword in meat_keywords if keyword in row['Name'].lower()]
    name_keywords_vegetarian = [keyword for keyword in vegetarian_keywords if keyword in row['Name'].lower()]
    
    recipe_keywords_meat = [keyword for keyword in meat_keywords if keyword in parse_ingredients(row['RecipeIngredientParts'].lower())]
    recipe_keywords_vegetarian = [keyword for keyword in vegetarian_keywords if keyword in parse_ingredients(row['RecipeIngredientParts'].lower())]

    contains_meat = bool(name_keywords_meat) or bool(recipe_keywords_meat)
    contains_non_vegan = bool(name_keywords_vegetarian) or bool(recipe_keywords_vegetarian)

    return pd.Series([int(contains_meat), int(not contains_meat and contains_non_vegan), int(not contains_non_vegan)])

    
# ----------------------------------------------------------- #
# IMPORTANT: function to encode the variables for the dataset #
# ----------------------------------------------------------- #
def encode_variables(df):
    # Like: object -> bool
    like_mapping = {False: 0, True: 1}
    df["Like"] = df["Like"].replace(like_mapping).astype('category')
    
    # HighProtein: {Indiferent, Yes} - object -> categorical
    hp_mapping = {'Indifferent': 0, 'Yes': 1}
    df["HighProtein"] = df["HighProtein"].replace(hp_mapping)
    df["HighProtein"] = df["HighProtein"].astype('category')
    
    # LowSugar: {0, Indiferent} - object -> categorical
    ls_mapping = {'0': 0, 'Indifferent': 1}
    df["LowSugar"] = df["LowSugar"].replace(ls_mapping)
    df["LowSugar"] = df["LowSugar"].astype('category')
    
    # Diet: {Vegetarian, Omnivore, Vegan} - object -> categorical
    df = dummify_var(df, ["Diet"])
    
    # Name: 140k values, 50% distinct - object -> categorical
    df[['MeatMeal', 'VegetarianMeal', 'VeganMeal']] = df.apply(create_products_columns, axis=1)
    # Drop the Name column
    df.drop(["Name"], axis=1, inplace=True)
    
    # RecipeCategory: 7 categories - object -> categorical
    df = dummify_var(df, ["RecipeCategory"])
    
    # RecipeYield: 46k values, 7.9k distinct, 93.8k missing - object -> categorical
    df.drop(["RecipeYield"], axis=1, inplace=True)
    
    # RecipeIngredientQuantities: compute sum and drop
    df["RecipeIngredientQuantitiesTotal"] = df["RecipeIngredientQuantities"].apply(sum_numeric_values)
    df["NumberOfRecipeIngredients"] = df["RecipeIngredientQuantities"].apply(lambda x: len(parse_ingredients(x)))
    df.drop(["RecipeIngredientQuantities"], axis=1, inplace=True)
    
    # Number of ingredients match
    df["MatchNumberOfIngredients"] = df.apply(lambda x: 1 if x["NumberOfRecipeIngredients"] == len(parse_ingredients(x["RecipeIngredientParts"])) else 0, axis=1)
    
    # RecipeIngredientParts: drop
    df.drop(["RecipeIngredientParts"], axis=1, inplace=True)
    
    # Drop ID columns 
    # AuthorId: 140k values, 35% distinct
    df.drop(["AuthorId"], axis=1, inplace=True)
    df.drop(["RecipeId"], axis=1, inplace=True)
    # Note: "Rating" was kept as float64 to allow for decimal values
    
    # New column: TotalTime
    df['TotalTime'] = df['CookTime'] + df['PrepTime']
    # New column: RespectsRequestedTime + drop Time
    df['RespectsRequestedTime'] = df.apply(lambda x: 1 if x['TotalTime'] <= x['Time'] else 0, axis=1)
    df['TimeDifference'] = df.apply(lambda x: x['TotalTime'] - x['Time'], axis=1)
    df.drop(["Time"], axis=1, inplace=True)
    
    # New colums for Age: <30, 30-60, >60
    #      Like         False     True 
    #      Age                               
    # (17.999, 30.0]  0.977451  0.022549
    # (30.0, 60.0]    0.884422  0.115578
    # (60.0, 79.0]    0.751216  0.248784
    df['Age_30'] = df['Age'].apply(lambda x: 1 if x < 30 else 0)
    df['Age_30_60'] = df['Age'].apply(lambda x: 1 if x >= 30 and x <= 60 else 0)
    df['Age_60'] = df['Age'].apply(lambda x: 1 if x > 60 else 0)

    return df


# encode the variables
print('Encoding variables...')
df = encode_variables(df)

df.to_csv(f'{file_dir}/{file_tag}_encoded.csv', index=True)


# In[40]:


# AUX: Get variable types
def get_variable_types(df):
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Categorical': []
    }

    for c in df.columns:
        if df[c].dtype == 'boolean':
            variable_types['Binary'].append(c)
        elif df[c].dtype == 'int8' or df[c].dtype == 'int16' or df[c].dtype == 'int32' or df[c].dtype == 'int64' or \
            df[c].dtype == 'float16' or df[c].dtype == 'float32' or df[c].dtype == 'float64':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'category' or df[c].dtype == 'object':
            variable_types['Categorical'].append(c)
        else:
            print(f'Unknown variable type for {c}')

    return variable_types


# #### Missing values

# In[41]:


file_source_path = 'dataset_encoded.csv' # source file
file_dir = '.' # destination directory
file_tag = 'dataset'

# Import data
df = pd.read_csv(file_source_path, low_memory=False)
index_column = df.columns[0]
df.drop([index_column], axis=1, inplace=True)

# --------------- #
# Missing Values  #
# --------------- #

print("Missing values:")
mv = {}
for var in df:
    nr = df[var].isna().sum()
    if nr > 0:
        mv[var] = nr
        print(f"{var} : {nr} ({round(nr/df[var].shape[0]*100, 2)}%)")


# In[42]:


# defines the number of records to discard entire COLUMNS
threshold = df.shape[0] * 0.90

# drop columns with more missing values than the defined threshold
missings = [c for c in mv.keys() if mv[c]>threshold]
df = df.drop(columns=missings, inplace=False)


# remove meaningless columns (after manual inspection) 
df = df.drop(columns=['Rating'], inplace=False)


# In[43]:


# --------------------------------------------------------------- #
# APPROACH 2: Fill with MOST FREQ Value after DROP Missing Values #
# --------------------------------------------------------------- #

# AUX: Fill with MOST FREQUENT value
def fill_with_most_frequent(data):
    tmp_nr, tmp_cat, tmp_bool = None, None, None
    variables = get_variable_types(data.drop(['TestSetId','Like'], axis=1))
    numeric_vars = variables['Numeric']
    categorical_vars = variables['Categorical']
    binary_vars = variables['Binary']

    tmp_nr, tmp_cat, tmp_bool = None, None, None
    if len(numeric_vars) > 0:
        imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
        tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
    if len(categorical_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_cat = DataFrame(imp.fit_transform(data[categorical_vars]), columns=categorical_vars)
    if len(binary_vars) > 0:
        imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
        tmp_bool = DataFrame(imp.fit_transform(data[binary_vars].astype(int)), columns=binary_vars).astype(bool)

    df = concat([tmp_nr, tmp_cat, tmp_bool], axis=1)
    df['TestSetId'] = data['TestSetId'] ; df['Like'] = data['Like']
    df.index = data.index

    return df

# ----------------------------------------------------------------- #

# Fill the rest with most frequent value
df_most_freq = fill_with_most_frequent(df)
df_most_freq.to_csv(f'{file_dir}/{file_tag}_drop_columns_then_most_frequent.csv', index=True)
# df_most_freq.head()


# ### Data Wrangling
# 
# In contrast to Data Cleaning, Data Wrangling _transforms_ the dataset, in order
# to prepare it for the training of the models. This includes scaling, dimensionality
# reduction, data augmentation, outlier removal, etc.

# #### Outliers treatment

# In[44]:


# Best option: dataset_drop_columns_then_most_frequent
file_source_path = 'dataset_drop_columns_then_most_frequent.csv' # source file
file_dir = '.' # destination directory
file_tag = 'dataset'

# read the data
df = pd.read_csv(file_source_path, low_memory=False)
index_column = df.columns[0]
df.drop([index_column], axis=1, inplace=True)

# print(df.info())

non_numeric_vars =  ['TestSetId', 'AuthorId', 'Diet', 'Name', 'RecipeCategory', 'RecipeIngredientQuantities', \
                     'RecipeIngredientParts', 'RecipeYield', 'HighCalories', 'HighProtein', 'LowFat', \
                     'LowSugar', 'HighFiber', 'Age_30', 'Age_30_60', 'Age_60', 'RespectsRequestedTime', \
                     'Diet_Omnivore', 'Diet_Vegan', 'Diet_Vegetarian', 'MeatMeal', 'VegetarianMeal', 'VeganMeal', \
                     'RecipeCategory_Beverages', 'RecipeCategory_Bread', 'RecipeCategory_Breakfast', \
                     'RecipeCategory_Lunch', 'RecipeCategory_One dish meal', 'RecipeCategory_Soup', 'RecipeCategory_Other', \
                     'MatchNumberOfIngredients', 'RespectsRequestedTime']

numeric_vars = get_variable_types(df)['Numeric']
# remove original non-numeric variables 
for var in numeric_vars.copy():
    if var in non_numeric_vars:
        numeric_vars.remove(var)

# Remove class (Like) from numerical variables
numeric_vars.remove('Like')

summary5 = df.describe(include='number')


# In[45]:


def determine_outlier_thresholds(summary5, var, OPTION, OUTLIER_PARAM):
    # default parameter
    if OPTION == 'iqr':
        iqr = OUTLIER_PARAM * (summary5[var]['75%'] - summary5[var]['25%'])
        top_threshold = summary5[var]['75%'] + iqr
        bottom_threshold = summary5[var]['25%'] - iqr
    # for normal distribution
    elif OPTION == 'stdev':
        std = OUTLIER_PARAM * summary5[var]['std']
        top_threshold = summary5[var]['mean'] + std
        bottom_threshold = summary5[var]['mean'] - std
    else:
        raise ValueError('Unknown outlier parameter!')
    return top_threshold, bottom_threshold


# In[46]:


# # ------------------------- #
# # APPROACH 1: Drop outliers #
# # ------------------------- #

# # Tuned parameter to get the better results
# IQR_PARAM = 2.5

# data = df.copy(deep=True)

# for var in numeric_vars:
#     top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM)
#     outliers = data[(data[var] > top_threshold) | (data[var] < bottom_threshold)]
#     # keep all the outliers that have a TestSetId
#     outliers = outliers[outliers['TestSetId'].isna()]
#     # print(f'{var} outliers: {outliers.shape[0]}/{data[var].shape[0]}')
#     data.drop(outliers.index, axis=0, inplace=True)
# data.to_csv(f'{file_tag}_drop_outliers_{IQR_PARAM}.csv', index=True)
# print('Dataset after dropping outliers:', data.shape)

# # Best results: Training score = 0.875

# # IQR_PARAM results:
# #    IQR    |    1    |   1.5   |    2    |   2.5   |    3    |   3.5   |    4    |
# # --------- |---------|---------|---------|---------|---------|---------|---------|
# #   Score   |  0.838  |  0.838  |  0.842  |  0.851  |  0.841  |  0.839  |  0.840  |


# In[47]:


# ----------------------------- #
# APPROACH 2: Truncate outliers #
# ----------------------------- #

# Tuned parameter to get the better results
IQR_PARAM = 1.5

data = df.copy(deep=True)

for var in numeric_vars:
    top_threshold, bottom_threshold = determine_outlier_thresholds(summary5, var, 'iqr', IQR_PARAM)
    # original_column = data[var].copy()
    data[var] = data[var].apply(lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x)
    # print(f'{var} outliers: {(data[var] != original_column).sum()}/{data[var].shape[0]}')
data.to_csv(f'{file_tag}_truncate_outliers_{IQR_PARAM}.csv', index=True)
print('Dataset after truncating outliers:', data.shape)
    
# Best results: Score = 0.842 -> IQR_PARAM = 1.5

# IQR_PARAM results:
#    IQR    |    1    |   1.5   |    2    |   2.5   |    3    |
# --------- |---------|---------|---------|---------|---------|
#   Score   |  0.842  |  0.842  |  0.834  |  0.834  |  0.831  |


# #### Scaling

# In[48]:


# Best option: dataset_truncate_outliers_1.5
file_source_path = 'dataset_truncate_outliers_1.5.csv' # source file
file_dir = '.' # destination directory
file_tag = 'dataset'

# read the data
df = pd.read_csv(file_source_path, low_memory=False)
index_column = df.columns[0]
df = df.drop([index_column], axis=1)

variable_types = get_variable_types(df)
numeric_vars = variable_types['Numeric']
categorical_vars = variable_types['Categorical']
boolean_vars = variable_types['Binary']
rest_vars = []

# print('Numeric variables:', numeric_vars)
# print('Categorical variables:', categorical_vars)
# print('Boolean variables:', boolean_vars)

non_numeric_vars =  ['TestSetId', 'AuthorId', 'Diet', 'Name', 'RecipeCategory', 'RecipeIngredientQuantities', \
                     'RecipeIngredientParts', 'RecipeYield', 'HighCalories', 'HighProtein', 'LowFat', \
                     'LowSugar', 'HighFiber', 'Age_30', 'Age_30_60', 'Age_60', 'RespectsRequestedTime', \
                     'Diet_Omnivore', 'Diet_Vegan', 'Diet_Vegetarian', 'MeatMeal', 'VegetarianMeal', 'VeganMeal', \
                     'RecipeCategory_Beverages', 'RecipeCategory_Bread', 'RecipeCategory_Breakfast', \
                     'RecipeCategory_Lunch', 'RecipeCategory_One dish meal', 'RecipeCategory_Soup', 'RecipeCategory_Other', \
                     'MatchNumberOfIngredients', 'RespectsRequestedTime']

# remove original non-numeric variables 
for var in numeric_vars.copy():
    if var in non_numeric_vars:
        numeric_vars.remove(var)
        rest_vars.append(var)

# Remove class (Like) from numerical variables
numeric_vars.remove('Like')

df_num = df[numeric_vars]
df_symb = df[categorical_vars]
df_bool = df[boolean_vars]
df_rest = df[rest_vars]
df_target = df['Like']


# **MinMax**

# In[49]:


# --------------------- #
# MinMax normalization  #
# --------------------- #

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_num)
tmp = DataFrame(transf.transform(df_num), index=df.index, columns= numeric_vars)
temp_norm_data_minmax = concat([tmp, df_symb, df_bool, df_rest], axis=1)
norm_data_minmax = concat([temp_norm_data_minmax, df_target], axis=1)
norm_data_minmax.to_csv(f'{file_tag}_scaled_minmax.csv', index=True)
# print(norm_data_minmax.describe())


# **Extra:** Summary Report

# In[50]:


# # Best option: dataset_rating_drop_recipe_mean
# file_source_path = 'datasets/scaling/dataset_dataset_drop_outliers_1.5_scaled_minmax.csv' # source file

# # read the data
# df = pd.read_csv(file_source_path, low_memory=False)
# convert_variable_types(df)
# index_column = df.columns[0]
# df.drop([index_column], axis=1, inplace=True)

# # We can also leverage the dataprep package to get a nice summary report
# report = sv.analyze(df)
# report.show_notebook()

# # We can also leverage the yadata_profiling package to get a nice summary report
# profile = ProfileReport(df, title="LLMeals - Summary Report")
# profile


# #### Feature Selection

# In[51]:


# Best option: dataset_scaled_minmax
file_source_path = 'dataset_scaled_minmax.csv' # source file
file_tag = 'dataset'

# read the data
df = pd.read_csv(file_source_path, low_memory=False)
index_column = df.columns[0]
df.drop([index_column], axis=1, inplace=True)

variable_types = get_variable_types(df)
numeric_vars = variable_types['Numeric']

# print('Numeric variables:', numeric_vars)

non_numeric_vars =  ['TestSetId', 'AuthorId', 'Diet', 'Name', 'RecipeCategory', 'RecipeIngredientQuantities', \
                     'RecipeIngredientParts', 'RecipeYield', 'HighCalories', 'HighProtein', 'LowFat', \
                     'LowSugar', 'HighFiber', 'Age_30', 'Age_30_60', 'Age_60', 'RespectsRequestedTime', \
                     'Diet_Omnivore', 'Diet_Vegan', 'Diet_Vegetarian', 'MeatMeal', 'VegetarianMeal', 'VeganMeal', \
                     'RecipeCategory_Beverages', 'RecipeCategory_Bread', 'RecipeCategory_Breakfast', \
                     'RecipeCategory_Lunch', 'RecipeCategory_One dish meal', 'RecipeCategory_Soup', 'RecipeCategory_Other', \
                     'MatchNumberOfIngredients', 'RespectsRequestedTime']

# remove original non-numeric variables 
for var in numeric_vars.copy():
    if var in non_numeric_vars:
        numeric_vars.remove(var)

# Remove class (Like) from categorical variables
numeric_vars.remove('Like')

df_num = df[numeric_vars]

# ---------------------------- #
# Dropping Redundant Variables #
# ---------------------------- #

THRESHOLD = 0.9

def select_redundant(corr_mtx, threshold):
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx

drop, corr_mtx = select_redundant(df_num.corr(), THRESHOLD)

if corr_mtx.empty:
    print('Matrix is empty. No redundant variables to drop.')

# figure(figsize=[12, 12])
# heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
# title('Filtered Correlation Analysis')
# tight_layout()
# show()


# In[52]:


def drop_redundant(data, vars_2drop):
    sel_2drop = []
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    # print('Variables to drop: ', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df

# print("Variables before drop: ", drop.keys())
# df = drop_redundant(df, drop)

df.to_csv(f'{file_tag}_selected.csv', index=True)


# ### Feature Extraction
# We experimented with applying PCA to the dataset to reduce its dimensionality while retaining as much of the original information as possible. However, instead of yielding better results, the outcomes worsened.

# ### Sampling

# Once we have a cleaned data set, and before we start the *Modelling* phase, we are going to split our data set into multiple sub-datasets. 
# Here, we are going to balance the data to ensure that both classes are equally represented, and then split it into an *train* and *test* data set.

# In[53]:


file_source_path = 'dataset_selected.csv' # source file
target = 'Like'

# read data
df = pd.read_csv(f'{file_source_path}', low_memory=False)
# remove index column
index_column = df.columns[0]
df = df.drop([index_column], axis=1)
# Drop TestSetId column and Like NaN rows (no way to know if they liked or not)
df.drop('TestSetId', axis=1, inplace=True)
df.dropna(subset=[target], inplace=True)

# print(df.shape)
# # Take a random sample of 10k rows
# df_test = df.sample(n=int(df.shape[0]/10)).copy(deep=True)
# df_test.to_csv(f'df_test.csv', index=False)
# df.drop(df_test.index, axis=0, inplace=True)
# print(df.shape)

# ----------------------------- #
#           BALANCING           #
# ----------------------------- #
like_zero = df[df[target] == 0.0]
like_one = df[df[target] == 1.0]

df_one_sample = like_one.sample(len(like_zero), replace=True)
df_zero_sample = like_zero.sample(len(like_zero))

df = pd.concat([df_zero_sample, df_one_sample], axis=0)
# ----------------------------- #

y = df.pop(target).values
X = df.values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                test_size=0.3, 
                shuffle=True,
                random_state=3)


# ### <font color='green'> Phase 4: Modeling </font>
# 
# In this phase, the model is trained and tuned. In general, data transformations
# from data wrangling can be part of a machine learning pipeline, and can therefore
# be tuned as well. (See CRISP-DM: DataPrep <--> Modeling)

# In[54]:


# Here, we want to find the best classifier. As candidates, we consider
#   1. LogisticRegression
#   2. RandomForestClassifier
#   3. GradientBoostingClassifier
#   4. HistGradientBoostingClassifier
#   5. AdaBoostClassifier
#   6. MLPClassifier
    
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


model_logistic_regression = LogisticRegression(max_iter=300)
model_random_forest = RandomForestClassifier()
model_gradient_boosting = GradientBoostingClassifier()
model_adaboost = AdaBoostClassifier()
model_hist_gradient_boosting = HistGradientBoostingClassifier()
model_neural_network = MLPClassifier(max_iter=200)


pipeline = Pipeline(steps=[("model", None)])

parameter_grid_preprocessing = {
  # Empty on purpose (no preprocessing)
}

# NOTE: Logistic Regression does not perform as well as the other models
parameter_grid_logistic_regression = {
  "model" : [model_logistic_regression],
  "model__C" : [0.1, 1, 10],  # inverse regularization strength
}

parameter_grid_gradient_boosting = {
  "model" : [model_gradient_boosting],
  "model__n_estimators" : [150],
  "model__max_depth" : list(range(df.shape[1]-12, df.shape[1]+1)), # 4
  # "model__learning_rate" : [0.2],
}

# This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).
parameter_grid_hist_gradient_boosting = {
  "model" : [model_hist_gradient_boosting],
  # "model__learning_rate" : [0.2], 
  "model__max_depth" : list(range(15, df.shape[1]+1)), # 4
}

parameter_grid_adaboost = {
  "model" : [model_adaboost],
  "model__n_estimators" : [150]
}

parameter_grid_random_forest = {
  "model" : [model_random_forest],
  "model__n_estimators" : [10, 20, 50],  # number of max trees in the forest
  "model__max_depth" : [5, 10, 15],
}

# NOTE: NN does not perform well on this dataset + takes a long time to train
parameter_grid_neural_network = {
  "model": [model_neural_network],
  "model__hidden_layer_sizes": [(30, 30), (40, 30)],  # Example hidden layer configurations
  "model__alpha": [0.0001],  # Regularization parameter
}


meta_parameter_grid = [
                      # parameter_grid_logistic_regression,
                      #  parameter_grid_random_forest] #,
                      #  parameter_grid_gradient_boosting] #,
                       parameter_grid_hist_gradient_boosting] #,
                      #  parameter_grid_adaboost ] #,
                      #  parameter_grid_neural_network]

meta_parameter_grid = [{**parameter_grid_preprocessing, **model_grid}
                       for model_grid in meta_parameter_grid]

search = GridSearchCV(pipeline,
                      meta_parameter_grid, 
                      scoring="balanced_accuracy",
                      n_jobs=-1, 
                      cv=5,  # number of folds for cross-validation 
                      error_score="raise"
)

# here, the actual training and grid search happens
search.fit(X_train, y_train.ravel())

print("best parameter:", search.best_params_ ,"(CV score=%0.3f)" % search.best_score_)


# ### <font color='green'> Step 5: Evaluation </font>
# 
# Once the appropriate models are chosen, they are evaluated on the test set. For
# this, different evaluation metrics can be used. Furthermore, this step is where
# the models and their predictions are analyzed resp. different properties, including
# feature importance, robustness to outliers, etc.

# In[55]:


# evaluate performance of model on test set
print("Score on test set:", search.score(X_test, y_test.ravel()))

# contingency table
ct = pd.crosstab(search.best_estimator_.predict(X_test), y_test.ravel(),
                 rownames=["pred"], colnames=["true"])
print(ct)


# In[56]:


# # for a detailed look on the performance of the different models (if different models are used)
# def get_search_score_overview():
#   for c,s in zip(search.cv_results_["params"],search.cv_results_["mean_test_score"]):
#       print(c, s)

# print(get_search_score_overview())


# ### <font color='green'> Step 6: Deployment </font>

# In[57]:


# # ----------------------------------------------------------------- #
# # TEST with a random sample of 10k rows from the original dataset   #
# # ----------------------------------------------------------------- #

# # read data
# df_test = pd.read_csv('df_test.csv', low_memory=False)

# def micro_service_classify_review_test(datapoint):
#   # make sure the provided datapoints adhere to the correct format for model input
  
#   # fetch your trained model
#   model = search.best_estimator_

#   # make prediction with the model
#   prediction = model.predict(datapoint)

#   return prediction

# # Save the Like values in a vector
# like_values = df_test['Like'].values

# # Optionally, you can drop the 'Like' column from the sampled_df if you don't need it
# df_test.drop('Like', axis=1, inplace=True)

# # make the missing predictions for the Like column
# df_test['Like'] = micro_service_classify_review_test(df_test.values)

# # Calculate balanced accuracy
# balanced_acc = balanced_accuracy_score(like_values, df_test['Like'])

# print(f"Balanced Accuracy: {balanced_acc}")


# In[58]:


def micro_service_classify_review(datapoint):
  # 'TestSetId' is not a feature used for prediction
  datapoint = datapoint.drop('TestSetId', axis=1)

  # fetch your trained model
  model = search.best_estimator_

  # make prediction with the model
  prediction = model.predict(datapoint)

  return prediction


# In the Analytics Cup, we need to export your prediction in a very specific output format. This is a csv file without an index and two columns, *id* and *prediction*. Note that the values in both columns need to be integer values, and especially in the *prediction* column either 1 or 0.

# In[59]:


file_source_path = 'dataset_selected.csv' # source file

# read data
df = pd.read_csv(f'{file_source_path}', low_memory=False)
# remove index column
index_column = df.columns[0]
df = df.drop([index_column], axis=1)

# keep only the rows without a Like value
df = df[df['Like'].isna()]
# remove the Like column
df.drop('Like', axis=1, inplace=True)

# make the missing predictions for the Like column
df['Like'] = micro_service_classify_review(df)

# create a dataset that contains only the column 
# with the TestSetId and the model prediction for Like
output = df[['TestSetId', 'Like']]

# rename the columns to match the required format
output = output.rename(columns={'TestSetId': 'id'})
output = output.rename(columns={'Like': 'prediction'})
submission = output.reindex(columns=["id", "prediction"])
# convert id and prediction to integer
submission['id'] = submission['id'].astype(int)
submission['prediction'] = submission['prediction'].astype(int)

# print(submission.head())

# save the submission to a CSV file
submission.to_csv('predictions_tugas.csv', index = False)
print("- Predictions saved to predictions_tugas.csv")

