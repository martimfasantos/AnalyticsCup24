import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Parse terminal input
FLAG = ''
valid_flags = ('special', 'missing_values', 'outliers', 'scaling', 'balancing', 'feature_selection')
if len(sys.argv) == 2 and sys.argv[1] in valid_flags:
    FLAG = sys.argv[1]
else:
    print("Invalid format, try:  python eval_aux.py [missing_values|outliers|scaling|balancing|feature_selection]")
    exit(1)
    
# Folder path
dir_path = f'datasets/{FLAG}/'

# List to store files
file_names = []
file_paths = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
        file_paths.append(f'datasets/{FLAG}/{file_name}')

print("Evaluating datasets: ", file_names)
print("-" * 20)

# ---------------------------------------------------------------------------

# target variable
target = 'Like'

# data scaling
transform_scaler = StandardScaler()

# dimensionality reduction
transform_pca = PCA()

for i in range(len(file_names)):
    file_path = file_paths[i]
    
    # read data
    df = pd.read_csv(f'{file_path}.csv', low_memory=False)
    # remove index column
    index_column = df.columns[0]
    df = df.drop([index_column], axis=1)

    # Drop TestSetId column and Like NaN rows (no way to know if they liked or not)
    df.drop('TestSetId', axis=1, inplace=True)
    df.dropna(subset=[target], inplace=True)
    
    # ----------------------------- #
    #           BALANCING           #
    # ----------------------------- #

    like_zero = df[df[target] == 0.0]
    like_one = df[df[target] == 1.0]

    df_one_sample = like_one.sample(len(like_zero), replace=True)
    df_zero_sample = like_zero.sample(len(like_zero))

    df = pd.concat([df_zero_sample, df_one_sample], axis=0)

    # print(df.info())
    # print(df[target].value_counts())

    y = df.pop(target).values
    X = df.values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                    test_size=0.3, 
                    shuffle=True,
                    random_state=3)
    
    model_logistic_regression = LogisticRegression(max_iter=300)
    model_random_forest = RandomForestClassifier()
    model_gradient_boosting = GradientBoostingClassifier()
    model_neural_network = MLPClassifier(max_iter=200)
    
    # train the models
    if FLAG == 'missing_values' or FLAG == 'outliers':
        pipeline = Pipeline(steps=[("scaler", transform_scaler), 
                                ("pca", transform_pca),
                                ("model", None)])
    else:
        pipeline = Pipeline(steps=[("pca", transform_pca),
                                ("model", None)])

    parameter_grid_preprocessing = {
    # "pca__n_components" : [1, 2, 3, 4],
        "pca__n_components" : [df.shape[1]-12, df.shape[1]-8, df.shape[1]-5, df.shape[1]]
    }

    # parameter_grid_logistic_regression = {
    #   "model" : [model_logistic_regression],
    #   "model__C" : [0.1, 1, 10],  # inverse regularization strength
    # }

    parameter_grid_gradient_boosting = {
        "model" : [model_gradient_boosting],
        "model__n_estimators" : [20, 30, 50]
    }

    parameter_grid_random_forest = {
        "model" : [model_random_forest],
        "model__n_estimators" : [30, 50, 75],  # number of max trees in the forest
        # "model__max_depth" : [2, 3, 4],
        "model__max_depth" : [20, df.shape[1]],
    }

    # NOTE: NN does not perform well on this dataset + takes a long time to train
    # parameter_grid_neural_network = {
    #     "model": [model_neural_network],
    #     "model__hidden_layer_sizes": [(30, 30), (40, 30)],  # Example hidden layer configurations
    #     "model__alpha": [0.0001],  # Regularization parameter
    # }
    
    
    meta_parameter_grid = [
                    # parameter_grid_logistic_regression,
                       parameter_grid_random_forest,
                       parameter_grid_gradient_boosting] #,
                      #  parameter_grid_neural_network]

    meta_parameter_grid = [{**parameter_grid_preprocessing, **model_grid}
                        for model_grid in meta_parameter_grid]

    search = GridSearchCV(pipeline,
                        meta_parameter_grid, 
                        scoring="balanced_accuracy",
                        # n_jobs=2, 
                        n_jobs=-1, 
                        cv=5,  # number of folds for cross-validation 
                        error_score="raise"
    )

    # here, the actual training and grid search happens
    search.fit(X_train, y_train.ravel())

    print(f"{file_names[i]}: train best parameters:", search.best_params_ ,"(CV score=%0.3f)" % search.best_score_)
    print("-" * 20)
    
    # evaluate performance of model on test set
    print("Score on test set:", search.score(X_test, y_test.ravel()))    

    # contingency table
    ct = pd.crosstab(search.best_estimator_.predict(X_test), y_test.ravel(),
                    rownames=["pred"], colnames=["true"])
    print(ct)
    print("-" * 20)